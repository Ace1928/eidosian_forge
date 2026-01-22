import datetime
import os
import time
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystoneauth1 import session
import oslo_cache
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import pbr.version
import testresources
from testtools import matchers
import webob
import webob.dec
from keystonemiddleware import auth_token
from keystonemiddleware.auth_token import _base
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as ksm_exceptions
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit import client_fixtures
class CommonAuthTokenMiddlewareTest(object):
    """These tests are run once using v2 tokens and again using v3 tokens."""

    def test_init_does_not_call_http(self):
        self.create_simple_middleware(conf={})
        self.assertLastPath(None)

    def test_auth_with_no_token_does_not_call_http(self):
        middleware = self.create_simple_middleware()
        self.call(middleware, expected_status=401)
        self.assertLastPath(None)

    def test_init_by_ipv6Addr_auth_host(self):
        del self.conf['identity_uri']
        conf = {'auth_host': '2001:2013:1:f101::1', 'auth_port': '1234', 'auth_protocol': 'http', 'www_authenticate_uri': None, 'auth_version': 'v3.0'}
        middleware = self.create_simple_middleware(conf=conf)
        self.assertEqual('http://[2001:2013:1:f101::1]:1234', middleware._www_authenticate_uri)

    def assert_valid_request_200(self, token, with_catalog=True):
        resp = self.call_middleware(headers={'X-Auth-Token': token})
        if with_catalog:
            self.assertTrue(resp.request.headers.get('X-Service-Catalog'))
        else:
            self.assertNotIn('X-Service-Catalog', resp.request.headers)
        self.assertEqual(FakeApp.SUCCESS, resp.body)
        self.assertIn('keystone.token_info', resp.request.environ)
        return resp.request

    def test_valid_uuid_request(self):
        for _ in range(2):
            token = self.token_dict['uuid_token_default']
            self.assert_valid_request_200(token)
            self.assert_valid_last_url(token)

    def test_valid_uuid_request_with_auth_fragments(self):
        del self.conf['identity_uri']
        self.conf['auth_protocol'] = 'https'
        self.conf['auth_host'] = 'keystone.example.com'
        self.conf['auth_port'] = '1234'
        self.conf['auth_admin_prefix'] = '/testadmin'
        self.set_middleware()
        self.assert_valid_request_200(self.token_dict['uuid_token_default'])
        self.assert_valid_last_url(self.token_dict['uuid_token_default'])

    def test_request_invalid_uuid_token(self):
        invalid_uri = '%s/v2.0/tokens/invalid-token' % BASE_URI
        self.requests_mock.get(invalid_uri, status_code=404)
        resp = self.call_middleware(headers={'X-Auth-Token': 'invalid-token'}, expected_status=401)
        self.assertEqual('Keystone uri="https://keystone.example.com:1234"', resp.headers['WWW-Authenticate'])

    def test_request_no_token(self):
        resp = self.call_middleware(expected_status=401)
        self.assertEqual('Keystone uri="https://keystone.example.com:1234"', resp.headers['WWW-Authenticate'])

    def test_request_no_token_http(self):
        resp = self.call_middleware(method='HEAD', expected_status=401)
        self.assertEqual('Keystone uri="https://keystone.example.com:1234"', resp.headers['WWW-Authenticate'])

    def test_request_blank_token(self):
        resp = self.call_middleware(headers={'X-Auth-Token': ''}, expected_status=401)
        self.assertEqual('Keystone uri="https://keystone.example.com:1234"', resp.headers['WWW-Authenticate'])

    def _get_cached_token(self, token):
        return self.middleware._token_cache.get(token)

    def test_memcache_set_invalid_uuid(self):
        invalid_uri = '%s/v3/tokens/invalid-token' % BASE_URI
        self.requests_mock.get(invalid_uri, status_code=404)
        token = 'invalid-token'
        self.call_middleware(headers={'X-Auth-Token': token}, expected_status=401)

    def test_memcache_set_expired(self, extra_conf={}, extra_environ={}):
        token_cache_time = 10
        conf = {'token_cache_time': '%s' % token_cache_time}
        conf.update(extra_conf)
        self.set_middleware(conf=conf)
        token = self.token_dict['uuid_token_default']
        self.call_middleware(headers={'X-Auth-Token': token})
        req = webob.Request.blank('/')
        req.headers['X-Auth-Token'] = token
        req.environ.update(extra_environ)
        now = datetime.datetime.now(datetime.timezone.utc)
        self.useFixture(TimeFixture(now))
        req.get_response(self.middleware)
        self.assertIsNotNone(self._get_cached_token(token))
        timeutils.advance_time_seconds(token_cache_time)
        self.assertIsNone(self._get_cached_token(token))

    def test_swift_memcache_set_expired(self):
        extra_conf = {'cache': 'swift.cache'}
        extra_environ = {'swift.cache': _cache._FakeClient()}
        self.test_memcache_set_expired(extra_conf, extra_environ)

    def test_http_error_not_cached_token(self):
        """Test to don't cache token as invalid on network errors.

        We use UUID tokens since they are the easiest one to reach
        get_http_connection.
        """
        self.set_middleware(conf={'http_request_max_retries': '0'})
        self.call_middleware(headers={'X-Auth-Token': ERROR_TOKEN}, expected_status=503)
        self.assertIsNone(self._get_cached_token(ERROR_TOKEN))
        self.assert_valid_last_url(ERROR_TOKEN)

    def test_discovery_failure(self):

        def discovery_failure_response(request, context):
            raise ksa_exceptions.DiscoveryFailure('Could not determine a suitable URL for the plugin')
        self.requests_mock.get(BASE_URI, text=discovery_failure_response)
        self.call_middleware(headers={'X-Auth-Token': 'token'}, expected_status=503)
        self.assertIsNone(self._get_cached_token('token'))
        self.assertEqual(BASE_URI, self.requests_mock.last_request.url)

    def test_http_request_max_retries(self):
        times_retry = 10
        body_string = 'The Keystone service is temporarily unavailable.'
        conf = {'http_request_max_retries': '%s' % times_retry}
        self.set_middleware(conf=conf)
        with mock.patch('time.sleep') as mock_obj:
            self.call_middleware(headers={'X-Auth-Token': ERROR_TOKEN}, expected_status=503, expected_body_string=body_string)
        self.assertEqual(mock_obj.call_count, times_retry)

    def test_request_timeout(self):
        self.call_middleware(headers={'X-Auth-Token': TIMEOUT_TOKEN}, expected_status=503)
        self.assertIsNone(self._get_cached_token(TIMEOUT_TOKEN))
        self.assert_valid_last_url(TIMEOUT_TOKEN)

    def test_nocatalog(self):
        conf = {'include_service_catalog': 'False'}
        self.set_middleware(conf=conf)
        self.assert_valid_request_200(self.token_dict['uuid_token_default'], with_catalog=False)

    def assert_kerberos_bind(self, token, bind_level, use_kerberos=True, success=True):
        conf = {'enforce_token_bind': bind_level, 'auth_version': self.auth_version}
        self.set_middleware(conf=conf)
        req = webob.Request.blank('/')
        req.headers['X-Auth-Token'] = token
        if use_kerberos:
            if use_kerberos is True:
                req.environ['REMOTE_USER'] = self.examples.KERBEROS_BIND
            else:
                req.environ['REMOTE_USER'] = use_kerberos
            req.environ['AUTH_TYPE'] = 'Negotiate'
        resp = req.get_response(self.middleware)
        if success:
            self.assertEqual(200, resp.status_int)
            self.assertEqual(FakeApp.SUCCESS, resp.body)
            self.assertIn('keystone.token_info', req.environ)
            self.assert_valid_last_url(token)
        else:
            self.assertEqual(401, resp.status_int)
            msg = 'Keystone uri="https://keystone.example.com:1234"'
            self.assertEqual(msg, resp.headers['WWW-Authenticate'])

    def test_uuid_bind_token_disabled_with_kerb_user(self):
        for use_kerberos in [True, False]:
            self.assert_kerberos_bind(self.token_dict['uuid_token_bind'], bind_level='disabled', use_kerberos=use_kerberos, success=True)

    def test_uuid_bind_token_disabled_with_incorrect_ticket(self):
        self.assert_kerberos_bind(self.token_dict['uuid_token_bind'], bind_level='kerberos', use_kerberos='ronald@MCDONALDS.COM', success=False)

    def test_uuid_bind_token_permissive_with_kerb_user(self):
        self.assert_kerberos_bind(self.token_dict['uuid_token_bind'], bind_level='permissive', use_kerberos=True, success=True)

    def test_uuid_bind_token_permissive_without_kerb_user(self):
        self.assert_kerberos_bind(self.token_dict['uuid_token_bind'], bind_level='permissive', use_kerberos=False, success=False)

    def test_uuid_bind_token_permissive_with_unknown_bind(self):
        token = self.token_dict['uuid_token_unknown_bind']
        for use_kerberos in [True, False]:
            self.assert_kerberos_bind(token, bind_level='permissive', use_kerberos=use_kerberos, success=True)

    def test_uuid_bind_token_permissive_with_incorrect_ticket(self):
        self.assert_kerberos_bind(self.token_dict['uuid_token_bind'], bind_level='kerberos', use_kerberos='ronald@MCDONALDS.COM', success=False)

    def test_uuid_bind_token_strict_with_kerb_user(self):
        self.assert_kerberos_bind(self.token_dict['uuid_token_bind'], bind_level='strict', use_kerberos=True, success=True)

    def test_uuid_bind_token_strict_with_kerbout_user(self):
        self.assert_kerberos_bind(self.token_dict['uuid_token_bind'], bind_level='strict', use_kerberos=False, success=False)

    def test_uuid_bind_token_strict_with_unknown_bind(self):
        token = self.token_dict['uuid_token_unknown_bind']
        for use_kerberos in [True, False]:
            self.assert_kerberos_bind(token, bind_level='strict', use_kerberos=use_kerberos, success=False)

    def test_uuid_bind_token_required_with_kerb_user(self):
        self.assert_kerberos_bind(self.token_dict['uuid_token_bind'], bind_level='required', use_kerberos=True, success=True)

    def test_uuid_bind_token_required_without_kerb_user(self):
        self.assert_kerberos_bind(self.token_dict['uuid_token_bind'], bind_level='required', use_kerberos=False, success=False)

    def test_uuid_bind_token_required_with_unknown_bind(self):
        token = self.token_dict['uuid_token_unknown_bind']
        for use_kerberos in [True, False]:
            self.assert_kerberos_bind(token, bind_level='required', use_kerberos=use_kerberos, success=False)

    def test_uuid_bind_token_required_without_bind(self):
        for use_kerberos in [True, False]:
            self.assert_kerberos_bind(self.token_dict['uuid_token_default'], bind_level='required', use_kerberos=use_kerberos, success=False)

    def test_uuid_bind_token_named_kerberos_with_kerb_user(self):
        self.assert_kerberos_bind(self.token_dict['uuid_token_bind'], bind_level='kerberos', use_kerberos=True, success=True)

    def test_uuid_bind_token_named_kerberos_without_kerb_user(self):
        self.assert_kerberos_bind(self.token_dict['uuid_token_bind'], bind_level='kerberos', use_kerberos=False, success=False)

    def test_uuid_bind_token_named_kerberos_with_unknown_bind(self):
        token = self.token_dict['uuid_token_unknown_bind']
        for use_kerberos in [True, False]:
            self.assert_kerberos_bind(token, bind_level='kerberos', use_kerberos=use_kerberos, success=False)

    def test_uuid_bind_token_named_kerberos_without_bind(self):
        for use_kerberos in [True, False]:
            self.assert_kerberos_bind(self.token_dict['uuid_token_default'], bind_level='kerberos', use_kerberos=use_kerberos, success=False)

    def test_uuid_bind_token_named_kerberos_with_incorrect_ticket(self):
        self.assert_kerberos_bind(self.token_dict['uuid_token_bind'], bind_level='kerberos', use_kerberos='ronald@MCDONALDS.COM', success=False)

    def test_uuid_bind_token_with_unknown_named_FOO(self):
        token = self.token_dict['uuid_token_bind']
        for use_kerberos in [True, False]:
            self.assert_kerberos_bind(token, bind_level='FOO', use_kerberos=use_kerberos, success=False)

    def test_caching_token_on_verify(self):
        self.middleware._token_cache._env_cache_name = 'cache'
        cache = _cache._FakeClient()
        self.middleware._token_cache.initialize(env={'cache': cache})
        orig_cache_set = cache.set
        cache.set = mock.Mock(side_effect=orig_cache_set)
        token = self.token_dict['uuid_token_default']
        self.call_middleware(headers={'X-Auth-Token': token})
        self.assertThat(1, matchers.Equals(cache.set.call_count))
        self.call_middleware(headers={'X-Auth-Token': token})
        self.assertThat(1, matchers.Equals(cache.set.call_count))

    def test_auth_plugin(self):
        for service_url in (self.examples.UNVERSIONED_SERVICE_URL, self.examples.SERVICE_URL):
            self.requests_mock.get(service_url, json=VERSION_LIST_v3, status_code=300)
        token = self.token_dict['uuid_token_default']
        resp = self.call_middleware(headers={'X-Auth-Token': token})
        self.assertEqual(FakeApp.SUCCESS, resp.body)
        token_auth = resp.request.environ['keystone.token_auth']
        endpoint_filter = {'service_type': self.examples.SERVICE_TYPE, 'version': 3}
        url = token_auth.get_endpoint(session.Session(), **endpoint_filter)
        self.assertEqual('%s/v3' % BASE_URI, url)
        self.assertTrue(token_auth.has_user_token)
        self.assertFalse(token_auth.has_service_token)
        self.assertIsNone(token_auth.service)

    def test_doesnt_auto_set_content_type(self):
        text = uuid.uuid4().hex

        def _middleware(environ, start_response):
            start_response(200, [])
            return text

        def _start_response(status_code, headerlist, exc_info=None):
            self.assertIn('200', status_code)
            self.assertEqual([], headerlist)
        m = auth_token.AuthProtocol(_middleware, self.conf)
        env = {'REQUEST_METHOD': 'GET', 'HTTP_X_AUTH_TOKEN': self.token_dict['uuid_token_default']}
        r = m(env, _start_response)
        self.assertEqual(text, r)

    def test_auth_plugin_service_token(self):
        url = 'http://test.url'
        text = uuid.uuid4().hex
        self.requests_mock.get(url, text=text)
        token = self.token_dict['uuid_token_default']
        resp = self.call_middleware(headers={'X-Auth-Token': token})
        self.assertEqual(200, resp.status_int)
        self.assertEqual(FakeApp.SUCCESS, resp.body)
        s = session.Session(auth=resp.request.environ['keystone.token_auth'])
        resp = s.get(url)
        self.assertEqual(text, resp.text)
        self.assertEqual(200, resp.status_code)
        headers = self.requests_mock.last_request.headers
        self.assertEqual(FAKE_ADMIN_TOKEN_ID, headers['X-Service-Token'])

    def test_service_token_with_valid_service_role_not_required(self):
        self.conf['service_token_roles'] = ['service']
        self.conf['service_token_roles_required'] = False
        self.set_middleware(conf=self.conf)
        user_token = self.token_dict['uuid_token_default']
        service_token = self.token_dict['uuid_service_token_default']
        resp = self.call_middleware(headers={'X-Auth-Token': user_token, 'X-Service-Token': service_token})
        self.assertEqual('Confirmed', resp.request.headers['X-Service-Identity-Status'])

    def test_service_token_with_invalid_service_role_not_required(self):
        self.conf['service_token_roles'] = [uuid.uuid4().hex]
        self.conf['service_token_roles_required'] = False
        self.set_middleware(conf=self.conf)
        user_token = self.token_dict['uuid_token_default']
        service_token = self.token_dict['uuid_service_token_default']
        resp = self.call_middleware(headers={'X-Auth-Token': user_token, 'X-Service-Token': service_token})
        self.assertEqual('Confirmed', resp.request.headers['X-Service-Identity-Status'])

    def test_service_token_with_valid_service_role_required(self):
        self.conf['service_token_roles'] = ['service']
        self.conf['service_token_roles_required'] = True
        self.set_middleware(conf=self.conf)
        user_token = self.token_dict['uuid_token_default']
        service_token = self.token_dict['uuid_service_token_default']
        resp = self.call_middleware(headers={'X-Auth-Token': user_token, 'X-Service-Token': service_token})
        self.assertEqual('Confirmed', resp.request.headers['X-Service-Identity-Status'])

    def test_service_token_with_invalid_service_role_required(self):
        self.conf['service_token_roles'] = [uuid.uuid4().hex]
        self.conf['service_token_roles_required'] = True
        self.set_middleware(conf=self.conf)
        user_token = self.token_dict['uuid_token_default']
        service_token = self.token_dict['uuid_service_token_default']
        resp = self.call_middleware(headers={'X-Auth-Token': user_token, 'X-Service-Token': service_token}, expected_status=401)
        self.assertEqual('Invalid', resp.request.headers['X-Service-Identity-Status'])