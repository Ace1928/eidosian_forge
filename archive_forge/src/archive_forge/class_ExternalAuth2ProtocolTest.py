import base64
import copy
import hashlib
import jwt.utils
import logging
import ssl
from testtools import matchers
import time
from unittest import mock
import uuid
import webob.dec
import fixtures
from oslo_config import cfg
import six
from six.moves import http_client
import testresources
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session
from keystonemiddleware.auth_token import _cache
from keystonemiddleware import external_oauth2_token
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit import client_fixtures
from keystonemiddleware.tests.unit import utils
class ExternalAuth2ProtocolTest(BaseExternalOauth2TokenMiddlewareTest):

    def setUp(self):
        super(ExternalAuth2ProtocolTest, self).setUp()
        self._test_client_id = str(uuid.uuid4())
        self._test_client_secret = str(uuid.uuid4())
        self._auth_method = 'client_secret_basic'
        self._test_conf = get_config(introspect_endpoint=self._introspect_endpoint, audience=self._audience, auth_method=self._auth_method, client_id=self._test_client_id, client_secret=self._test_client_secret, thumbprint_verify=False, mapping_project_id='access_project.id', mapping_project_name='access_project.name', mapping_project_domain_id='access_project.domain.id', mapping_project_domain_name='access_project.domain.name', mapping_user_id='client_id', mapping_user_name='username', mapping_user_domain_id='user_domain.id', mapping_user_domain_name='user_domain.name', mapping_roles='roles', mapping_system_scope='system.all', mapping_expires_at='exp', memcached_servers=','.join(MEMCACHED_SERVERS), memcache_use_advanced_pool=True, memcache_pool_dead_retry=300, memcache_pool_maxsize=10, memcache_pool_unused_timeout=60, memcache_pool_conn_get_timeout=10, memcache_pool_socket_timeout=3, memcache_security_strategy=None, memcache_secret_key=None)
        uuid_token_default = self.examples.v3_UUID_TOKEN_DEFAULT
        uuid_serv_token_default = self.examples.v3_UUID_SERVICE_TOKEN_DEFAULT
        uuid_token_bind = self.examples.v3_UUID_TOKEN_BIND
        uuid_service_token_bind = self.examples.v3_UUID_SERVICE_TOKEN_BIND
        self.token_dict = {'uuid_token_default': uuid_token_default, 'uuid_service_token_default': uuid_serv_token_default, 'uuid_token_bind': uuid_token_bind, 'uuid_service_token_bind': uuid_service_token_bind}
        self._token = self.token_dict['uuid_token_default']
        self._user_id = str(uuid.uuid4()) + '_user_id'
        self._user_name = str(uuid.uuid4()) + '_user_name'
        self._user_domain_id = str(uuid.uuid4()) + '_user_domain_id'
        self._user_domain_name = str(uuid.uuid4()) + '_user_domain_name'
        self._project_id = str(uuid.uuid4()) + '_project_id'
        self._project_name = str(uuid.uuid4()) + '_project_name'
        self._project_domain_id = str(uuid.uuid4()) + 'project_domain_id'
        self._project_domain_name = str(uuid.uuid4()) + 'project_domain_name'
        self._roles = 'admin,member,reader'
        self._default_metadata = {'access_project': {'id': self._project_id, 'name': self._project_name, 'domain': {'id': self._project_domain_id, 'name': self._project_domain_name}}, 'user_domain': {'id': self._user_domain_id, 'name': self._user_domain_name}, 'roles': self._roles, 'client_id': self._user_id, 'username': self._user_name, 'exp': int(time.time()) + 3600}
        self._clear_call_count = 0
        cert = self.examples.V3_OAUTH2_MTLS_CERTIFICATE
        self._pem_client_cert = cert.decode('ascii')
        self._der_client_cert = ssl.PEM_cert_to_DER_cert(self._pem_client_cert)
        thumb_sha256 = hashlib.sha256(self._der_client_cert).digest()
        self._cert_thumb = jwt.utils.base64url_encode(thumb_sha256).decode('ascii')

    def test_token_cache_factory_insecure(self):
        conf = copy.deepcopy(self._test_conf)
        self.set_middleware(conf=conf)
        self.assertIsInstance(self.middleware._token_cache, _cache.TokenCache)

    def test_token_cache_factory_secure(self):
        conf = copy.deepcopy(self._test_conf)
        conf['memcache_secret_key'] = 'test_key'
        conf['memcache_security_strategy'] = 'MAC'
        self.set_middleware(conf=conf)
        self.assertIsInstance(self.middleware._token_cache, _cache.SecureTokenCache)
        conf['memcache_security_strategy'] = 'ENCRYPT'
        self.set_middleware(conf=conf)
        self.assertIsInstance(self.middleware._token_cache, _cache.SecureTokenCache)

    def test_caching_token_on_verify(self):
        conf = copy.deepcopy(self._test_conf)
        self.set_middleware(conf=conf)
        self.middleware._token_cache._env_cache_name = 'cache'
        cache = _cache._FakeClient()
        self.middleware._token_cache.initialize(env={'cache': cache})
        orig_cache_set = cache.set
        cache.set = mock.Mock(side_effect=orig_cache_set)

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=self._default_metadata)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
        self.call_middleware(headers=get_authorization_header(self._token), expected_status=200, method='GET', path='/vnfpkgm/v1/vnf_packages', der_client_cert=self._der_client_cert, environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})
        self.assertThat(1, matchers.Equals(cache.set.call_count))
        self.call_middleware(headers=get_authorization_header(self._token), expected_status=200, method='GET', path='/vnfpkgm/v1/vnf_packages', der_client_cert=self._der_client_cert, environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})
        self.assertThat(1, matchers.Equals(cache.set.call_count))

    def test_caching_token_timeout(self):
        conf = copy.deepcopy(self._test_conf)
        self.set_middleware(conf=conf)
        self.middleware._token_cache._env_cache_name = 'cache'
        cache = _cache._FakeClient()
        self.middleware._token_cache.initialize(env={'cache': cache})
        self._default_metadata['exp'] = int(time.time()) - 3600
        orig_cache_set = cache.set
        cache.set = mock.Mock(side_effect=orig_cache_set)

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=self._default_metadata)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
        self.call_middleware(headers=get_authorization_header(self._token), expected_status=200, method='GET', path='/vnfpkgm/v1/vnf_packages', der_client_cert=self._der_client_cert, environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})
        self.assertThat(1, matchers.Equals(cache.set.call_count))
        self.call_middleware(headers=get_authorization_header(self._token), expected_status=401, method='GET', path='/vnfpkgm/v1/vnf_packages', der_client_cert=self._der_client_cert, environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})

    @mock.patch('keystonemiddleware.auth_token._cache.TokenCache.get')
    def test_caching_token_type_invalid(self, mock_cache_get):
        mock_cache_get.return_value = 'test'
        conf = copy.deepcopy(self._test_conf)
        self.set_middleware(conf=conf)
        self.middleware._token_cache._env_cache_name = 'cache'
        cache = _cache._FakeClient()
        self.middleware._token_cache.initialize(env={'cache': cache})
        orig_cache_set = cache.set
        cache.set = mock.Mock(side_effect=orig_cache_set)

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=self._default_metadata)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
        self.call_middleware(headers=get_authorization_header(self._token), expected_status=401, method='GET', path='/vnfpkgm/v1/vnf_packages', der_client_cert=self._der_client_cert, environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})

    def test_caching_token_not_active(self):
        conf = copy.deepcopy(self._test_conf)
        self.set_middleware(conf=conf)
        self.middleware._token_cache._env_cache_name = 'cache'
        cache = _cache._FakeClient()
        self.middleware._token_cache.initialize(env={'cache': cache})
        orig_cache_set = cache.set
        cache.set = mock.Mock(side_effect=orig_cache_set)

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=False, metadata=self._default_metadata)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
        self.call_middleware(headers=get_authorization_header(self._token), expected_status=401, method='GET', path='/vnfpkgm/v1/vnf_packages', der_client_cert=self._der_client_cert, environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})
        self.assertThat(1, matchers.Equals(cache.set.call_count))
        self.call_middleware(headers=get_authorization_header(self._token), expected_status=401, method='GET', path='/vnfpkgm/v1/vnf_packages', der_client_cert=self._der_client_cert, environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})
        self.assertThat(1, matchers.Equals(cache.set.call_count))

    def test_caching_token_invalid(self):
        conf = copy.deepcopy(self._test_conf)
        self.set_middleware(conf=conf)
        self.middleware._token_cache._env_cache_name = 'cache'
        cache = _cache._FakeClient()
        self.middleware._token_cache.initialize(env={'cache': cache})
        orig_cache_set = cache.set
        cache.set = mock.Mock(side_effect=orig_cache_set)

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=self._default_metadata)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
        self.call_middleware(headers=get_authorization_header(self._token), expected_status=200, method='GET', path='/vnfpkgm/v1/vnf_packages', der_client_cert=self._der_client_cert, environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})
        self.assertThat(1, matchers.Equals(cache.set.call_count))
        self.call_middleware(headers=get_authorization_header(str(uuid.uuid4()) + '_token'), expected_status=500, method='GET', path='/vnfpkgm/v1/vnf_packages', der_client_cert=self._der_client_cert, environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})
        self._token = self.token_dict['uuid_token_default']