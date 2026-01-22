from __future__ import annotations
import functools
import inspect
import ipaddress
import json
import mimetypes
import os
import re
import types
import warnings
from http.client import responses
from logging import Logger
from typing import TYPE_CHECKING, Any, Awaitable, Coroutine, Sequence, cast
from urllib.parse import urlparse
import prometheus_client
from jinja2 import TemplateNotFound
from jupyter_core.paths import is_hidden
from jupyter_events import EventLogger
from tornado import web
from tornado.log import app_log
from traitlets.config import Application
import jupyter_server
from jupyter_server import CallContext
from jupyter_server._sysinfo import get_sys_info
from jupyter_server._tz import utcnow
from jupyter_server.auth.decorator import allow_unauthenticated, authorized
from jupyter_server.auth.identity import User
from jupyter_server.i18n import combine_translations
from jupyter_server.services.security import csp_report_uri
from jupyter_server.utils import (
class JupyterHandler(AuthenticatedHandler):
    """Jupyter-specific extensions to authenticated handling

    Mostly property shortcuts to Jupyter-specific settings.
    """

    @property
    def config(self) -> dict[str, Any] | None:
        return cast('dict[str, Any] | None', self.settings.get('config', None))

    @property
    def log(self) -> Logger:
        """use the Jupyter log by default, falling back on tornado's logger"""
        return log()

    @property
    def jinja_template_vars(self) -> dict[str, Any]:
        """User-supplied values to supply to jinja templates."""
        return cast('dict[str, Any]', self.settings.get('jinja_template_vars', {}))

    @property
    def serverapp(self) -> ServerApp | None:
        return cast('ServerApp | None', self.settings['serverapp'])

    @property
    def version_hash(self) -> str:
        """The version hash to use for cache hints for static files"""
        return cast(str, self.settings.get('version_hash', ''))

    @property
    def mathjax_url(self) -> str:
        url = cast(str, self.settings.get('mathjax_url', ''))
        if not url or url_is_absolute(url):
            return url
        return url_path_join(self.base_url, url)

    @property
    def mathjax_config(self) -> str:
        return cast(str, self.settings.get('mathjax_config', 'TeX-AMS-MML_HTMLorMML-full,Safe'))

    @property
    def default_url(self) -> str:
        return cast(str, self.settings.get('default_url', ''))

    @property
    def ws_url(self) -> str:
        return cast(str, self.settings.get('websocket_url', ''))

    @property
    def contents_js_source(self) -> str:
        self.log.debug('Using contents: %s', self.settings.get('contents_js_source', 'services/contents'))
        return cast(str, self.settings.get('contents_js_source', 'services/contents'))

    @property
    def kernel_manager(self) -> AsyncMappingKernelManager:
        return cast('AsyncMappingKernelManager', self.settings['kernel_manager'])

    @property
    def contents_manager(self) -> ContentsManager:
        return cast('ContentsManager', self.settings['contents_manager'])

    @property
    def session_manager(self) -> SessionManager:
        return cast('SessionManager', self.settings['session_manager'])

    @property
    def terminal_manager(self) -> TerminalManager:
        return cast('TerminalManager', self.settings['terminal_manager'])

    @property
    def kernel_spec_manager(self) -> KernelSpecManager:
        return cast('KernelSpecManager', self.settings['kernel_spec_manager'])

    @property
    def config_manager(self) -> ConfigManager:
        return cast('ConfigManager', self.settings['config_manager'])

    @property
    def event_logger(self) -> EventLogger:
        return cast('EventLogger', self.settings['event_logger'])

    @property
    def allow_origin(self) -> str:
        """Normal Access-Control-Allow-Origin"""
        return cast(str, self.settings.get('allow_origin', ''))

    @property
    def allow_origin_pat(self) -> str | None:
        """Regular expression version of allow_origin"""
        return cast('str | None', self.settings.get('allow_origin_pat', None))

    @property
    def allow_credentials(self) -> bool:
        """Whether to set Access-Control-Allow-Credentials"""
        return cast(bool, self.settings.get('allow_credentials', False))

    def set_default_headers(self) -> None:
        """Add CORS headers, if defined"""
        super().set_default_headers()

    def set_cors_headers(self) -> None:
        """Add CORS headers, if defined

        Now that current_user is async (jupyter-server 2.0),
        must be called at the end of prepare(), instead of in set_default_headers.
        """
        if self.allow_origin:
            self.set_header('Access-Control-Allow-Origin', self.allow_origin)
        elif self.allow_origin_pat:
            origin = self.get_origin()
            if origin and re.match(self.allow_origin_pat, origin):
                self.set_header('Access-Control-Allow-Origin', origin)
        elif self.token_authenticated and 'Access-Control-Allow-Origin' not in self.settings.get('headers', {}):
            self.set_header('Access-Control-Allow-Origin', self.request.headers.get('Origin', ''))
        if self.allow_credentials:
            self.set_header('Access-Control-Allow-Credentials', 'true')

    def set_attachment_header(self, filename: str) -> None:
        """Set Content-Disposition: attachment header

        As a method to ensure handling of filename encoding
        """
        escaped_filename = url_escape(filename)
        self.set_header('Content-Disposition', f"attachment; filename*=utf-8''{escaped_filename}")

    def get_origin(self) -> str | None:
        if 'Origin' in self.request.headers:
            origin = self.request.headers.get('Origin')
        else:
            origin = self.request.headers.get('Sec-Websocket-Origin', None)
        return origin

    def check_origin(self, origin_to_satisfy_tornado: str='') -> bool:
        """Check Origin for cross-site API requests, including websockets

        Copied from WebSocket with changes:

        - allow unspecified host/origin (e.g. scripts)
        - allow token-authenticated requests
        """
        if self.allow_origin == '*' or self.skip_check_origin():
            return True
        host = self.request.headers.get('Host')
        origin = self.request.headers.get('Origin')
        if origin is None or host is None:
            return True
        origin = origin.lower()
        origin_host = urlparse(origin).netloc
        if origin_host == host:
            return True
        if self.allow_origin:
            allow = bool(self.allow_origin == origin)
        elif self.allow_origin_pat:
            allow = bool(re.match(self.allow_origin_pat, origin))
        else:
            allow = False
        if not allow:
            self.log.warning('Blocking Cross Origin API request for %s.  Origin: %s, Host: %s', self.request.path, origin, host)
        return allow

    def check_referer(self) -> bool:
        """Check Referer for cross-site requests.
        Disables requests to certain endpoints with
        external or missing Referer.
        If set, allow_origin settings are applied to the Referer
        to whitelist specific cross-origin sites.
        Used on GET for api endpoints and /files/
        to block cross-site inclusion (XSSI).
        """
        if self.allow_origin == '*' or self.skip_check_origin():
            return True
        host = self.request.headers.get('Host')
        referer = self.request.headers.get('Referer')
        if not host:
            self.log.warning('Blocking request with no host')
            return False
        if not referer:
            self.log.warning('Blocking request with no referer')
            return False
        referer_url = urlparse(referer)
        referer_host = referer_url.netloc
        if referer_host == host:
            return True
        origin = f'{referer_url.scheme}://{referer_url.netloc}'
        if self.allow_origin:
            allow = self.allow_origin == origin
        elif self.allow_origin_pat:
            allow = bool(re.match(self.allow_origin_pat, origin))
        else:
            allow = False
        if not allow:
            self.log.warning('Blocking Cross Origin request for %s.  Referer: %s, Host: %s', self.request.path, origin, host)
        return allow

    def check_xsrf_cookie(self) -> None:
        """Bypass xsrf cookie checks when token-authenticated"""
        if not hasattr(self, '_jupyter_current_user'):
            return None
        if self.token_authenticated or self.settings.get('disable_check_xsrf', False):
            return None
        try:
            return super().check_xsrf_cookie()
        except web.HTTPError as e:
            if self.request.method in {'GET', 'HEAD'}:
                if not self.check_referer():
                    referer = self.request.headers.get('Referer')
                    if referer:
                        msg = f'Blocking Cross Origin request from {referer}.'
                    else:
                        msg = 'Blocking request from unknown origin'
                    raise web.HTTPError(403, msg) from e
            else:
                raise

    def check_host(self) -> bool:
        """Check the host header if remote access disallowed.

        Returns True if the request should continue, False otherwise.
        """
        if self.settings.get('allow_remote_access', False):
            return True
        match = re.match('^(.*?)(:\\d+)?$', self.request.host)
        assert match is not None
        host = match.group(1)
        if host.startswith('[') and host.endswith(']'):
            host = host[1:-1]
        check_host = urldecode_unix_socket_path(host)
        if check_host.startswith('/') and os.path.exists(check_host):
            allow = True
        else:
            try:
                addr = ipaddress.ip_address(host)
            except ValueError:
                allow = host in self.settings.get('local_hostnames', ['localhost'])
            else:
                allow = addr.is_loopback
        if not allow:
            self.log.warning("Blocking request with non-local 'Host' %s (%s). If the server should be accessible at that name, set ServerApp.allow_remote_access to disable the check.", host, self.request.host)
        return allow

    async def prepare(self, *, _redirect_to_login=True) -> Awaitable[None] | None:
        """Prepare a response."""
        CallContext.set(CallContext.JUPYTER_HANDLER, self)
        if not self.check_host():
            self.current_user = self._jupyter_current_user = None
            raise web.HTTPError(403)
        from jupyter_server.auth import IdentityProvider
        mod_obj = inspect.getmodule(self.get_current_user)
        assert mod_obj is not None
        user: User | None = None
        if type(self.identity_provider) is IdentityProvider and mod_obj.__name__ != __name__:
            warnings.warn('Overriding JupyterHandler.get_current_user is deprecated in jupyter-server 2.0. Use an IdentityProvider class.', DeprecationWarning, stacklevel=1)
            user = User(self.get_current_user())
        else:
            _user = self.identity_provider.get_user(self)
            if isinstance(_user, Awaitable):
                _user = await _user
            user = _user
        self.current_user = self._jupyter_current_user = user
        self.set_cors_headers()
        if self.request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            self.check_xsrf_cookie()
        if not self.settings.get('allow_unauthenticated_access', False):
            if not self.request.method:
                raise HTTPError(403)
            method = getattr(self, self.request.method.lower())
            if not getattr(method, '__allow_unauthenticated', False):
                if _redirect_to_login:
                    return web.authenticated(lambda _: super().prepare())(self)
                else:
                    user = self.current_user
                    if user is None:
                        self.log.warning(f"Couldn't authenticate {self.__class__.__name__} connection")
                        raise web.HTTPError(403)
        return super().prepare()

    def get_template(self, name):
        """Return the jinja template object for a given name"""
        return self.settings['jinja2_env'].get_template(name)

    def render_template(self, name, **ns):
        """Render a template by name."""
        ns.update(self.template_namespace)
        template = self.get_template(name)
        return template.render(**ns)

    @property
    def template_namespace(self) -> dict[str, Any]:
        return dict(base_url=self.base_url, default_url=self.default_url, ws_url=self.ws_url, logged_in=self.logged_in, allow_password_change=getattr(self.identity_provider, 'allow_password_change', False), auth_enabled=self.identity_provider.auth_enabled, login_available=self.identity_provider.login_available, token_available=bool(self.token), static_url=self.static_url, sys_info=json_sys_info(), contents_js_source=self.contents_js_source, version_hash=self.version_hash, xsrf_form_html=self.xsrf_form_html, token=self.token, xsrf_token=self.xsrf_token.decode('utf8'), nbjs_translations=json.dumps(combine_translations(self.request.headers.get('Accept-Language', ''))), **self.jinja_template_vars)

    def get_json_body(self) -> dict[str, Any] | None:
        """Return the body of the request as JSON data."""
        if not self.request.body:
            return None
        body = self.request.body.strip().decode('utf-8')
        try:
            model = json.loads(body)
        except Exception as e:
            self.log.debug('Bad JSON: %r', body)
            self.log.error("Couldn't parse JSON", exc_info=True)
            raise web.HTTPError(400, 'Invalid JSON in body of request') from e
        return cast('dict[str, Any]', model)

    def write_error(self, status_code: int, **kwargs: Any) -> None:
        """render custom error pages"""
        exc_info = kwargs.get('exc_info')
        message = ''
        status_message = responses.get(status_code, 'Unknown HTTP Error')
        if exc_info:
            exception = exc_info[1]
            try:
                message = exception.log_message % exception.args
            except Exception:
                pass
            reason = getattr(exception, 'reason', '')
            if reason:
                status_message = reason
        else:
            exception = '(unknown)'
        ns = {'status_code': status_code, 'status_message': status_message, 'message': message, 'exception': exception}
        self.set_header('Content-Type', 'text/html')
        try:
            html = self.render_template('%s.html' % status_code, **ns)
        except TemplateNotFound:
            html = self.render_template('error.html', **ns)
        self.write(html)