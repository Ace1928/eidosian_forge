import base64
import collections
import hashlib
import io
import json
import re
import textwrap
import time
from urllib import parse as urlparse
import zipfile
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import auth_context_middleware
from tensorboard.backend import client_feature_flags
from tensorboard.backend import empty_path_redirect
from tensorboard.backend import experiment_id
from tensorboard.backend import experimental_plugin
from tensorboard.backend import http_util
from tensorboard.backend import path_prefix
from tensorboard.backend import security_validator
from tensorboard.plugins import base_plugin
from tensorboard.plugins.core import core_plugin
from tensorboard.util import tb_logging
class TensorBoardWSGI:
    """The TensorBoard WSGI app that delegates to a set of TBPlugin."""

    def __init__(self, plugins, path_prefix='', data_provider=None, experimental_plugins=None, auth_providers=None, experimental_middlewares=None):
        """Constructs TensorBoardWSGI instance.

        Args:
          plugins: A list of base_plugin.TBPlugin subclass instances.
          path_prefix: A prefix of the path when app isn't served from root.
          data_provider: `tensorboard.data.provider.DataProvider` or
            `None`; if present, will inform the "active" state of
            `/plugins_listing`.
          experimental_plugins: A list of plugin names that are only provided
              experimentally. The corresponding plugins will only be activated for
              a user if the user has specified the plugin with the experimentalPlugin
              query parameter in the URL.
          auth_providers: Optional mapping whose values are `AuthProvider`
            values and whose keys are used by (e.g.) data providers to specify
            `AuthProvider`s via the `AuthContext.get` interface.
            Defaults to `{}`.
          experimental_middlewares: Optional list of WSGI middlewares to apply
            directly around the core TensorBoard app itself. Defaults to `[]`.
            This parameter is experimental and may be reworked or removed.

        Returns:
          A WSGI application for the set of all TBPlugin instances.

        Raises:
          ValueError: If some plugin has no plugin_name
          ValueError: If some plugin has an invalid plugin_name (plugin
              names must only contain [A-Za-z0-9_.-])
          ValueError: If two plugins have the same plugin_name
          ValueError: If some plugin handles a route that does not start
              with a slash

        :type plugins: list[base_plugin.TBPlugin]
        """
        self._plugins = plugins
        self._path_prefix = path_prefix
        self._data_provider = data_provider
        self._experimental_plugins = frozenset(experimental_plugins or ())
        self._auth_providers = auth_providers or {}
        self._extra_middlewares = list(experimental_middlewares or [])
        if self._path_prefix.endswith('/'):
            raise ValueError('Trailing slash in path prefix: %r' % self._path_prefix)
        self.exact_routes = {DATA_PREFIX + PLUGINS_LISTING_ROUTE: self._serve_plugins_listing, DATA_PREFIX + PLUGIN_ENTRY_ROUTE: self._serve_plugin_entry}
        unordered_prefix_routes = {}
        plugin_names_encountered = set()
        for plugin in self._plugins:
            if plugin.plugin_name is None:
                raise ValueError('Plugin %s has no plugin_name' % plugin)
            if not _VALID_PLUGIN_RE.match(plugin.plugin_name):
                raise ValueError('Plugin %s has invalid name %r' % (plugin, plugin.plugin_name))
            if plugin.plugin_name in plugin_names_encountered:
                raise ValueError('Duplicate plugins for name %s' % plugin.plugin_name)
            plugin_names_encountered.add(plugin.plugin_name)
            try:
                plugin_apps = plugin.get_plugin_apps()
            except Exception as e:
                if type(plugin) is core_plugin.CorePlugin:
                    raise
                logger.warning('Plugin %s failed. Exception: %s', plugin.plugin_name, str(e))
                continue
            for route, app in plugin_apps.items():
                if not route.startswith('/'):
                    raise ValueError('Plugin named %r handles invalid route %r: route does not start with a slash' % (plugin.plugin_name, route))
                if type(plugin) is core_plugin.CorePlugin:
                    path = route
                else:
                    path = DATA_PREFIX + PLUGIN_PREFIX + '/' + plugin.plugin_name + route
                if path.endswith('/*'):
                    path = path[:-1]
                    if '*' in path:
                        raise ValueError("Plugin %r handles invalid route '%s*': Only trailing wildcards are supported (i.e., `/.../*`)" % (plugin.plugin_name, path))
                    unordered_prefix_routes[path] = app
                else:
                    if '*' in path:
                        raise ValueError('Plugin %r handles invalid route %r: Only trailing wildcards are supported (i.e., `/.../*`)' % (plugin.plugin_name, path))
                    self.exact_routes[path] = app
        self.prefix_routes = collections.OrderedDict(sorted(unordered_prefix_routes.items(), key=lambda x: len(x[0]), reverse=True))
        self._app = self._create_wsgi_app()

    def _create_wsgi_app(self):
        """Apply middleware to create the final WSGI app."""
        app = self._route_request
        for middleware in self._extra_middlewares:
            app = middleware(app)
        app = auth_context_middleware.AuthContextMiddleware(app, self._auth_providers)
        app = client_feature_flags.ClientFeatureFlagsMiddleware(app)
        app = empty_path_redirect.EmptyPathRedirectMiddleware(app)
        app = experiment_id.ExperimentIdMiddleware(app)
        app = path_prefix.PathPrefixMiddleware(app, self._path_prefix)
        app = security_validator.SecurityValidatorMiddleware(app)
        app = _handling_errors(app)
        return app

    @wrappers.Request.application
    def _serve_plugin_entry(self, request):
        """Serves a HTML for iframed plugin entry point.

        Args:
          request: The werkzeug.Request object.

        Returns:
          A werkzeug.Response object.
        """
        name = request.args.get('name')
        plugins = [plugin for plugin in self._plugins if plugin.plugin_name == name]
        if not plugins:
            raise errors.NotFoundError(name)
        if len(plugins) > 1:
            reason = 'Plugin invariant error: multiple plugins with name {name} found: {list}'.format(name=name, list=plugins)
            raise AssertionError(reason)
        plugin = plugins[0]
        module_path = plugin.frontend_metadata().es_module_path
        if not module_path:
            return http_util.Respond(request, 'Plugin is not module loadable', 'text/plain', code=400)
        if urlparse.urlparse(module_path).netloc:
            raise ValueError('Expected es_module_path to be non-absolute path')
        module_json = json.dumps('.' + module_path)
        script_content = 'import({}).then((m) => void m.render());'.format(module_json)
        digest = hashlib.sha256(script_content.encode('utf-8')).digest()
        script_sha = base64.b64encode(digest).decode('ascii')
        html = textwrap.dedent('\n            <!DOCTYPE html>\n            <head><base href="plugin/{name}/" /></head>\n            <body><script type="module">{script_content}</script></body>\n            ').format(name=name, script_content=script_content)
        return http_util.Respond(request, html, 'text/html', csp_scripts_sha256s=[script_sha])

    @wrappers.Request.application
    def _serve_plugins_listing(self, request):
        """Serves an object mapping plugin name to whether it is enabled.

        Args:
          request: The werkzeug.Request object.

        Returns:
          A werkzeug.Response object.
        """
        response = collections.OrderedDict()
        ctx = plugin_util.context(request.environ)
        eid = plugin_util.experiment_id(request.environ)
        plugins_with_data = frozenset(self._data_provider.list_plugins(ctx, experiment_id=eid) or frozenset() if self._data_provider is not None else frozenset())
        plugins_to_skip = self._experimental_plugins - frozenset(request.args.getlist(EXPERIMENTAL_PLUGINS_QUERY_PARAM))
        for plugin in self._plugins:
            if plugin.plugin_name in plugins_to_skip:
                continue
            if type(plugin) is core_plugin.CorePlugin:
                continue
            is_active = bool(frozenset(plugin.data_plugin_names()) & plugins_with_data)
            if not is_active:
                try:
                    start = time.time()
                    is_active = plugin.is_active()
                    elapsed = time.time() - start
                    logger.info('Plugin listing: is_active() for %s took %0.3f seconds', plugin.plugin_name, elapsed)
                except Exception:
                    is_active = False
                    logger.error('Plugin listing: is_active() for %s failed (marking inactive)', plugin.plugin_name, exc_info=True)
            plugin_metadata = plugin.frontend_metadata()
            output_metadata = {'disable_reload': plugin_metadata.disable_reload, 'enabled': is_active, 'remove_dom': plugin_metadata.remove_dom}
            if plugin_metadata.tab_name is not None:
                output_metadata['tab_name'] = plugin_metadata.tab_name
            else:
                output_metadata['tab_name'] = plugin.plugin_name
            es_module_handler = plugin_metadata.es_module_path
            element_name = plugin_metadata.element_name
            is_ng_component = plugin_metadata.is_ng_component
            if is_ng_component:
                if element_name is not None:
                    raise ValueError('Plugin %r declared as both Angular built-in and legacy' % plugin.plugin_name)
                if es_module_handler is not None:
                    raise ValueError('Plugin %r declared as both Angular built-in and iframed' % plugin.plugin_name)
                loading_mechanism = {'type': 'NG_COMPONENT'}
            elif element_name is not None and es_module_handler is not None:
                logger.error('Plugin %r declared as both legacy and iframed; skipping', plugin.plugin_name)
                continue
            elif element_name is not None and es_module_handler is None:
                loading_mechanism = {'type': 'CUSTOM_ELEMENT', 'element_name': element_name}
            elif element_name is None and es_module_handler is not None:
                loading_mechanism = {'type': 'IFRAME', 'module_path': ''.join([request.script_root, DATA_PREFIX, PLUGIN_PREFIX, '/', plugin.plugin_name, es_module_handler])}
            else:
                loading_mechanism = {'type': 'NONE'}
            output_metadata['loading_mechanism'] = loading_mechanism
            response[plugin.plugin_name] = output_metadata
        return http_util.Respond(request, response, 'application/json')

    def __call__(self, environ, start_response):
        """Central entry point for the TensorBoard application.

        This __call__ method conforms to the WSGI spec, so that instances of this
        class are WSGI applications.

        Args:
          environ: See WSGI spec (PEP 3333).
          start_response: See WSGI spec (PEP 3333).
        """
        return self._app(environ, start_response)

    def _route_request(self, environ, start_response):
        """Delegate an incoming request to sub-applications.

        This method supports strict string matching and wildcard routes of a
        single path component, such as `/foo/*`. Other routing patterns,
        like regular expressions, are not supported.

        This is the main TensorBoard entry point before middleware is
        applied. (See `_create_wsgi_app`.)

        Args:
          environ: See WSGI spec (PEP 3333).
          start_response: See WSGI spec (PEP 3333).
        """
        request = wrappers.Request(environ)
        parsed_url = urlparse.urlparse(request.path)
        clean_path = _clean_path(parsed_url.path)
        if clean_path in self.exact_routes:
            return self.exact_routes[clean_path](environ, start_response)
        else:
            for path_prefix in self.prefix_routes:
                if clean_path.startswith(path_prefix):
                    return self.prefix_routes[path_prefix](environ, start_response)
            logger.warning('path %s not found, sending 404', clean_path)
            return http_util.Respond(request, 'Not found', 'text/plain', code=404)(environ, start_response)