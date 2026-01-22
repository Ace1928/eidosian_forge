import abc
import copy
import hashlib
import os
import ssl
import time
import uuid
import jwt.utils
import oslo_cache
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
import requests.auth
import webob.dec
import webob.exc
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from keystoneauth1.loading import session as session_loading
from keystonemiddleware._common import config
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.exceptions import ConfigurationError
from keystonemiddleware.exceptions import KeystoneMiddlewareException
from keystonemiddleware.i18n import _
class ExternalAuth2Protocol(object):
    """Middleware that handles External Server OAuth2.0 authentication."""

    def __init__(self, application, conf):
        super(ExternalAuth2Protocol, self).__init__()
        self._application = application
        self._log = logging.getLogger(conf.get('log_name', __name__))
        self._log.info('Starting Keystone external_oauth2_token middleware')
        config_opts = [(_EXT_AUTH_CONFIG_GROUP_NAME, _EXTERNAL_AUTH2_OPTS + loading.get_auth_common_conf_options())]
        all_opts = [(g, copy.deepcopy(o)) for g, o in config_opts]
        self._conf = config.Config('external_oauth2_token', _EXT_AUTH_CONFIG_GROUP_NAME, all_opts, conf)
        self._token_cache = self._token_cache_factory()
        self._session = self._create_session()
        self._audience = self._get_config_option('audience', is_required=True)
        self._introspect_endpoint = self._get_config_option('introspect_endpoint', is_required=True)
        self._auth_method = self._get_config_option('auth_method', is_required=True)
        self._client_id = self._get_config_option('client_id', is_required=True)
        self._http_client = _get_http_client(self._auth_method, self._session, self._introspect_endpoint, self._audience, self._client_id, self._get_config_option, self._log)

    def _token_cache_factory(self):
        security_strategy = self._conf.get('memcache_security_strategy')
        cache_kwargs = dict(cache_time=int(self._conf.get('token_cache_time')), memcached_servers=self._conf.get('memcached_servers'), use_advanced_pool=self._conf.get('memcache_use_advanced_pool'), dead_retry=self._conf.get('memcache_pool_dead_retry'), maxsize=self._conf.get('memcache_pool_maxsize'), unused_timeout=self._conf.get('memcache_pool_unused_timeout'), conn_get_timeout=self._conf.get('memcache_pool_conn_get_timeout'), socket_timeout=self._conf.get('memcache_pool_socket_timeout'))
        if security_strategy.lower() != 'none':
            secret_key = self._conf.get('memcache_secret_key')
            return _cache.SecureTokenCache(self._log, security_strategy, secret_key, **cache_kwargs)
        return _cache.TokenCache(self._log, **cache_kwargs)

    @webob.dec.wsgify()
    def __call__(self, req):
        """Handle incoming request."""
        self.process_request(req)
        response = req.get_response(self._application)
        return self.process_response(response)

    def process_request(self, request):
        """Process request.

        :param request: Incoming request
        :type request: _request.AuthTokenRequest
        """
        access_token = None
        if request.authorization and request.authorization.authtype == 'Bearer':
            access_token = request.authorization.params
        try:
            if not access_token:
                self._log.info('Unable to obtain the access token.')
                raise InvalidToken(_('Unable to obtain the access token.'))
            self._token_cache.initialize(request.environ)
            token_data = self._fetch_token(access_token)
            if self._get_config_option('thumbprint_verify', is_required=False):
                self._confirm_certificate_thumbprint(request, token_data.get('origin_token_metadata'))
            self._set_request_env(request, token_data)
        except InvalidToken as error:
            self._log.info('Rejecting request. Need a valid OAuth 2.0 access token. error: %s', error)
            message = _('The request you have made is denied, because the token is invalid.')
            body = {'error': {'code': 401, 'title': 'Unauthorized', 'message': message}}
            raise webob.exc.HTTPUnauthorized(body=jsonutils.dumps(body), headers=self._reject_headers, charset='UTF-8', content_type='application/json')
        except ForbiddenToken as error:
            self._log.warning('Rejecting request. The necessary information is required.error: %s', error)
            message = _('The request you have made is denied, because the necessary information could not be parsed.')
            body = {'error': {'code': 403, 'title': 'Forbidden', 'message': message}}
            raise webob.exc.HTTPForbidden(body=jsonutils.dumps(body), charset='UTF-8', content_type='application/json')
        except ConfigurationError as error:
            self._log.critical('Rejecting request. The configuration parameters are incorrect. error: %s', error)
            message = _('The request you have made is denied, because the configuration parameters are incorrect and the token can not be verified.')
            body = {'error': {'code': 500, 'title': 'Internal Server Error', 'message': message}}
            raise webob.exc.HTTPServerError(body=jsonutils.dumps(body), charset='UTF-8', content_type='application/json')
        except ServiceError as error:
            self._log.warning('Rejecting request. An exception occurred and the OAuth 2.0 access token can not be verified. error: %s', error)
            message = _('The request you have made is denied, because an exception occurred while accessing the external authentication server for token validation.')
            body = {'error': {'code': 500, 'title': 'Internal Server Error', 'message': message}}
            raise webob.exc.HTTPServerError(body=jsonutils.dumps(body), charset='UTF-8', content_type='application/json')

    def process_response(self, response):
        """Process Response.

        Add ``WWW-Authenticate`` headers to requests that failed with
        ``401 Unauthenticated`` so users know where to authenticate for future
        requests.
        """
        if response.status_int == 401:
            response.headers.extend(self._reject_headers)
        return response

    def _create_session(self, **kwargs):
        """Create session for HTTP access."""
        kwargs.setdefault('cert', self._get_config_option('certfile', is_required=False))
        kwargs.setdefault('key', self._get_config_option('keyfile', is_required=False))
        kwargs.setdefault('cacert', self._get_config_option('cafile', is_required=False))
        kwargs.setdefault('insecure', self._get_config_option('insecure', is_required=False))
        kwargs.setdefault('timeout', self._get_config_option('http_connect_timeout', is_required=False))
        kwargs.setdefault('user_agent', self._conf.user_agent)
        return session_loading.Session().load_from_options(**kwargs)

    def _get_config_option(self, key, is_required):
        """Read the value from config file by the config key."""
        value = self._conf.get(key)
        if not value:
            if is_required:
                self._log.critical('The value is required for option %s in group [%s]' % (key, _EXT_AUTH_CONFIG_GROUP_NAME))
                raise ConfigurationError(_('Configuration error. The parameter is not set for "%s" in group [%s].') % (key, _EXT_AUTH_CONFIG_GROUP_NAME))
            else:
                return None
        else:
            return value

    @property
    def _reject_headers(self):
        """Generate WWW-Authenticate Header.

        When response status is 401, this method will be called to add
        the 'WWW-Authenticate' header to the response.
        """
        header_val = 'Authorization OAuth 2.0 uri="%s"' % self._audience
        return [('WWW-Authenticate', header_val)]

    def _fetch_token(self, access_token):
        """Use access_token to get the valid token meta_data.

        Verify the access token through accessing the external
        authorization server.
        """
        try:
            cached = self._token_cache.get(access_token)
            if cached:
                self._log.debug('The cached token: %s' % cached)
                if not isinstance(cached, dict) or 'origin_token_metadata' not in cached:
                    self._log.warning('The cached data is invalid. %s' % cached)
                    raise InvalidToken(_('The token is invalid.'))
                origin_token_metadata = cached.get('origin_token_metadata')
                if not origin_token_metadata.get('active'):
                    self._log.warning('The cached data is invalid. %s' % cached)
                    raise InvalidToken(_('The token is invalid.'))
                expire_at = self._read_data_from_token(origin_token_metadata, 'mapping_expires_at', is_required=False, value_type=int)
                if expire_at:
                    if int(expire_at) < int(time.time()):
                        cached['origin_token_metadata']['active'] = False
                        self._token_cache.set(access_token, cached)
                        self._log.warning('The cached data is invalid. %s' % cached)
                        raise InvalidToken(_('The token is invalid.'))
                return cached
            http_response = self._http_client.introspect(access_token)
            if http_response.status_code != 200:
                self._log.critical('The introspect API returns an incorrect response. response_status: %s, response_text: %s' % (http_response.status_code, http_response.text))
                raise ServiceError(_('The token cannot be verified for validity.'))
            origin_token_metadata = http_response.json()
            self._log.debug('The introspect API response: %s' % origin_token_metadata)
            if not origin_token_metadata.get('active'):
                self._token_cache.set(access_token, {'origin_token_metadata': origin_token_metadata})
                self._log.info('The token is invalid. response: %s' % origin_token_metadata)
                raise InvalidToken(_('The token is invalid.'))
            token_data = self._parse_necessary_info(origin_token_metadata)
            self._token_cache.set(access_token, token_data)
            return token_data
        except (ConfigurationError, ForbiddenToken, ServiceError, InvalidToken):
            raise
        except (ksa_exceptions.ConnectFailure, ksa_exceptions.DiscoveryFailure, ksa_exceptions.RequestTimeout) as error:
            self._log.critical('Unable to validate token: %s', error)
            raise ServiceError(_('The Introspect API service is temporarily unavailable.'))
        except Exception as error:
            self._log.critical('Unable to validate token: %s', error)
            raise ServiceError(_('An exception occurred during the token verification process.'))

    def _read_data_from_token(self, token_metadata, config_key, is_required=False, value_type=None):
        """Read value from token metadata.

        Read the necessary information from the token metadata with the
        config key.
        """
        if not value_type:
            value_type = str
        meta_key = self._get_config_option(config_key, is_required=is_required)
        if not meta_key:
            return None
        if meta_key.find('.') >= 0:
            meta_value = None
            for temp_key in meta_key.split('.'):
                if not temp_key:
                    self._log.critical('Configuration error. config_key: %s , meta_key: %s ' % (config_key, meta_key))
                    raise ConfigurationError(_('Failed to parse the necessary information for the field "%s".') % meta_key)
                if not meta_value:
                    meta_value = token_metadata.get(temp_key)
                else:
                    if not isinstance(meta_value, dict):
                        self._log.warning('Failed to parse the necessary information. The meta_value is not of type dict.config_key: %s , meta_key: %s, value: %s' % (config_key, meta_key, meta_value))
                        raise ForbiddenToken(_('Failed to parse the necessary information for the field "%s".') % meta_key)
                    meta_value = meta_value.get(temp_key)
        else:
            meta_value = token_metadata.get(meta_key)
        if not meta_value:
            if is_required:
                self._log.warning('Failed to parse the necessary information. The meta value is required.config_key: %s , meta_key: %s, value: %s, need_type: %s' % (config_key, meta_key, meta_value, value_type))
                raise ForbiddenToken(_('Failed to parse the necessary information for the field "%s".') % meta_key)
            else:
                meta_value = None
        elif not isinstance(meta_value, value_type):
            self._log.warning('Failed to parse the necessary information. The meta value is of an incorrect type.config_key: %s , meta_key: %s, value: %s, need_type: %s' % (config_key, meta_key, meta_value, value_type))
            raise ForbiddenToken(_('Failed to parse the necessary information for the field "%s".') % meta_key)
        return meta_value

    def _parse_necessary_info(self, token_metadata):
        """Parse the necessary information from the token metadata."""
        token_data = dict()
        token_data['origin_token_metadata'] = token_metadata
        roles = self._read_data_from_token(token_metadata, 'mapping_roles', is_required=True)
        is_admin = 'false'
        if 'admin' in roles.lower().split(','):
            is_admin = 'true'
        token_data['roles'] = roles
        token_data['is_admin'] = is_admin
        system_scope = self._read_data_from_token(token_metadata, 'mapping_system_scope', is_required=False, value_type=bool)
        if system_scope:
            token_data['system_scope'] = 'all'
        else:
            project_id = self._read_data_from_token(token_metadata, 'mapping_project_id', is_required=False)
            if project_id:
                token_data['project_id'] = project_id
                token_data['project_name'] = self._read_data_from_token(token_metadata, 'mapping_project_name', is_required=True)
                token_data['project_domain_id'] = self._read_data_from_token(token_metadata, 'mapping_project_domain_id', is_required=True)
                token_data['project_domain_name'] = self._read_data_from_token(token_metadata, 'mapping_project_domain_name', is_required=True)
            else:
                token_data['domain_id'] = self._read_data_from_token(token_metadata, 'mapping_project_domain_id', is_required=True)
                token_data['domain_name'] = self._read_data_from_token(token_metadata, 'mapping_project_domain_name', is_required=True)
        token_data['user_id'] = self._read_data_from_token(token_metadata, 'mapping_user_id', is_required=True)
        token_data['user_name'] = self._read_data_from_token(token_metadata, 'mapping_user_name', is_required=True)
        token_data['user_domain_id'] = self._read_data_from_token(token_metadata, 'mapping_user_domain_id', is_required=True)
        token_data['user_domain_name'] = self._read_data_from_token(token_metadata, 'mapping_user_domain_name', is_required=True)
        return token_data

    def _get_client_certificate(self, request):
        """Get the client certificate from request environ or socket."""
        try:
            pem_client_cert = request.environ.get('SSL_CLIENT_CERT')
            if pem_client_cert:
                peer_cert = ssl.PEM_cert_to_DER_cert(pem_client_cert)
            else:
                wsgi_input = request.environ.get('wsgi.input')
                if not wsgi_input:
                    self._log.warn('Unable to obtain the client certificate. The object for wsgi_input is none.')
                    raise InvalidToken(_('Unable to obtain the client certificate.'))
                socket = wsgi_input.get_socket()
                if not socket:
                    self._log.warn('Unable to obtain the client certificate. The object for socket is none.')
                    raise InvalidToken(_('Unable to obtain the client certificate.'))
                peer_cert = socket.getpeercert(binary_form=True)
            if not peer_cert:
                self._log.warn('Unable to obtain the client certificate. The object for peer_cert is none.')
                raise InvalidToken(_('Unable to obtain the client certificate.'))
            return peer_cert
        except InvalidToken:
            raise
        except Exception as error:
            self._log.warn('Unable to obtain the client certificate. %s' % error)
            raise InvalidToken(_('Unable to obtain the client certificate.'))

    def _confirm_certificate_thumbprint(self, request, origin_token_metadata):
        """Check if the thumbprint in the token is valid."""
        peer_cert = self._get_client_certificate(request)
        try:
            thumb_sha256 = hashlib.sha256(peer_cert).digest()
            cert_thumb = jwt.utils.base64url_encode(thumb_sha256).decode('ascii')
        except Exception as error:
            self._log.warn('An Exception occurred. %s' % error)
            raise InvalidToken(_('Can not generate the thumbprint.'))
        token_thumb = origin_token_metadata.get('cnf', {}).get('x5t#S256')
        if cert_thumb != token_thumb:
            self._log.warn('The two thumbprints do not match. token_thumbprint: %s, certificate_thumbprint %s' % (token_thumb, cert_thumb))
            raise InvalidToken(_('The two thumbprints do not match.'))

    def _set_request_env(self, request, token_data):
        """Set request.environ with the necessary information."""
        request.environ['external.token_info'] = token_data
        request.environ['HTTP_X_IDENTITY_STATUS'] = 'Confirmed'
        request.environ['HTTP_X_ROLES'] = token_data.get('roles')
        request.environ['HTTP_X_ROLE'] = token_data.get('roles')
        request.environ['HTTP_X_USER_ID'] = token_data.get('user_id')
        request.environ['HTTP_X_USER_NAME'] = token_data.get('user_name')
        request.environ['HTTP_X_USER_DOMAIN_ID'] = token_data.get('user_domain_id')
        request.environ['HTTP_X_USER_DOMAIN_NAME'] = token_data.get('user_domain_name')
        if token_data.get('is_admin') == 'true':
            request.environ['HTTP_X_IS_ADMIN_PROJECT'] = token_data.get('is_admin')
        request.environ['HTTP_X_USER'] = token_data.get('user_name')
        if token_data.get('system_scope'):
            request.environ['HTTP_OPENSTACK_SYSTEM_SCOPE'] = token_data.get('system_scope')
        elif token_data.get('project_id'):
            request.environ['HTTP_X_PROJECT_ID'] = token_data.get('project_id')
            request.environ['HTTP_X_PROJECT_NAME'] = token_data.get('project_name')
            request.environ['HTTP_X_PROJECT_DOMAIN_ID'] = token_data.get('project_domain_id')
            request.environ['HTTP_X_PROJECT_DOMAIN_NAME'] = token_data.get('project_domain_name')
            request.environ['HTTP_X_TENANT_ID'] = token_data.get('project_id')
            request.environ['HTTP_X_TENANT_NAME'] = token_data.get('project_name')
            request.environ['HTTP_X_TENANT'] = token_data.get('project_id')
        else:
            request.environ['HTTP_X_DOMAIN_ID'] = token_data.get('domain_id')
            request.environ['HTTP_X_DOMAIN_NAME'] = token_data.get('domain_name')
        self._log.debug('The access token data is %s.' % jsonutils.dumps(token_data))