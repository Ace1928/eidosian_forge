import sys
import datetime
from unittest.mock import Mock
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.common.openstack import OpenStackBaseConnection
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import OpenStack_1_0_NodeDriver
from libcloud.test.compute.test_openstack import (
class OpenStackIdentityConnectionTestCase(unittest.TestCase):

    def setUp(self):
        OpenStackBaseConnection.auth_url = None
        OpenStackBaseConnection.conn_class = OpenStackMockHttp
        OpenStack_2_0_MockHttp.type = None
        OpenStackIdentity_3_0_MockHttp.type = None

    def test_auth_url_is_correctly_assembled(self):
        tuples = [('1.0', OpenStackMockHttp, {}), ('1.1', OpenStackMockHttp, {}), ('2.0', OpenStack_2_0_MockHttp, {}), ('2.0_apikey', OpenStack_2_0_MockHttp, {}), ('2.0_password', OpenStack_2_0_MockHttp, {}), ('3.x_password', OpenStackIdentity_3_0_MockHttp, {'tenant_name': 'tenant-name'}), ('3.x_appcred', OpenStackIdentity_3_0_MockHttp, {}), ('3.x_oidc_access_token', OpenStackIdentity_3_0_MockHttp, {'tenant_name': 'tenant-name'})]
        auth_urls = [('https://auth.api.example.com', ''), ('https://auth.api.example.com/', '/'), ('https://auth.api.example.com/foo/bar', '/foo/bar'), ('https://auth.api.example.com/foo/bar/', '/foo/bar/')]
        actions = {'1.0': '{url_path}/v1.0', '1.1': '{url_path}/v1.1/auth', '2.0': '{url_path}/v2.0/tokens', '2.0_apikey': '{url_path}/v2.0/tokens', '2.0_password': '{url_path}/v2.0/tokens', '3.x_password': '{url_path}/v3/auth/tokens', '3.x_appcred': '{url_path}/v3/auth/tokens', '3.x_oidc_access_token': '{url_path}/v3/OS-FEDERATION/identity_providers/user_name/protocols/tenant-name/auth'}
        user_id = OPENSTACK_PARAMS[0]
        key = OPENSTACK_PARAMS[1]
        for auth_version, mock_http_class, kwargs in tuples:
            for url, url_path in auth_urls:
                connection = self._get_mock_connection(mock_http_class=mock_http_class, auth_url=url)
                auth_url = connection.auth_url
                cls = get_class_for_auth_version(auth_version=auth_version)
                osa = cls(auth_url=auth_url, user_id=user_id, key=key, parent_conn=connection, **kwargs)
                try:
                    osa = osa.authenticate()
                except Exception:
                    pass
                expected_path = actions[auth_version].format(url_path=url_path).replace('//', '/')
                self.assertEqual(osa.action, expected_path)

    def test_basic_authentication(self):
        tuples = [('1.0', OpenStackMockHttp, {}), ('1.1', OpenStackMockHttp, {}), ('2.0', OpenStack_2_0_MockHttp, {}), ('2.0_apikey', OpenStack_2_0_MockHttp, {}), ('2.0_password', OpenStack_2_0_MockHttp, {}), ('3.x_password', OpenStackIdentity_3_0_MockHttp, {'user_id': 'test_user_id', 'key': 'test_key', 'token_scope': 'project', 'tenant_name': 'test_tenant', 'tenant_domain_id': 'test_tenant_domain_id', 'domain_name': 'test_domain'}), ('3.x_appcred', OpenStackIdentity_3_0_MockHttp, {'user_id': 'appcred_id', 'key': 'appcred_secret'}), ('3.x_oidc_access_token', OpenStackIdentity_3_0_MockHttp, {'user_id': 'test_user_id', 'key': 'test_key', 'token_scope': 'domain', 'tenant_name': 'test_tenant', 'tenant_domain_id': 'test_tenant_domain_id', 'domain_name': 'test_domain'})]
        user_id = OPENSTACK_PARAMS[0]
        key = OPENSTACK_PARAMS[1]
        for auth_version, mock_http_class, kwargs in tuples:
            connection = self._get_mock_connection(mock_http_class=mock_http_class)
            auth_url = connection.auth_url
            if not kwargs:
                kwargs['user_id'] = user_id
                kwargs['key'] = key
            cls = get_class_for_auth_version(auth_version=auth_version)
            osa = cls(auth_url=auth_url, parent_conn=connection, **kwargs)
            self.assertEqual(osa.urls, {})
            self.assertIsNone(osa.auth_token)
            self.assertIsNone(osa.auth_user_info)
            osa = osa.authenticate()
            self.assertTrue(len(osa.urls) >= 1)
            self.assertTrue(osa.auth_token is not None)
            if auth_version in ['1.1', '2.0', '2.0_apikey', '2.0_password', '3.x_password', '3.x_appcred', '3.x_oidc_access_token']:
                self.assertTrue(osa.auth_token_expires is not None)
            if auth_version in ['2.0', '2.0_apikey', '2.0_password', '3.x_password', '3.x_appcred', '3.x_oidc_access_token']:
                self.assertTrue(osa.auth_user_info is not None)

    def test_token_expiration_and_force_reauthentication(self):
        user_id = OPENSTACK_PARAMS[0]
        key = OPENSTACK_PARAMS[1]
        connection = self._get_mock_connection(OpenStack_2_0_MockHttp)
        auth_url = connection.auth_url
        osa = OpenStackIdentity_2_0_Connection(auth_url=auth_url, user_id=user_id, key=key, parent_conn=connection)
        mocked_auth_method = Mock(wraps=osa._authenticate_2_0_with_body)
        osa._authenticate_2_0_with_body = mocked_auth_method
        osa.auth_token = None
        osa.auth_token_expires = YESTERDAY
        count = 5
        for i in range(0, count):
            osa.authenticate(force=True)
        self.assertEqual(mocked_auth_method.call_count, count)
        osa.auth_token = None
        osa.auth_token_expires = YESTERDAY
        mocked_auth_method.call_count = 0
        self.assertEqual(mocked_auth_method.call_count, 0)
        for i in range(0, count):
            osa.authenticate(force=False)
        self.assertEqual(mocked_auth_method.call_count, 1)
        osa.auth_token = None
        mocked_auth_method.call_count = 0
        self.assertEqual(mocked_auth_method.call_count, 0)
        for i in range(0, count):
            osa.authenticate(force=False)
            if i == 0:
                osa.auth_token_expires = TOMORROW
        self.assertEqual(mocked_auth_method.call_count, 1)
        soon = datetime.datetime.utcnow() + datetime.timedelta(seconds=AUTH_TOKEN_EXPIRES_GRACE_SECONDS - 1)
        osa.auth_token = None
        mocked_auth_method.call_count = 0
        self.assertEqual(mocked_auth_method.call_count, 0)
        for i in range(0, count):
            if i == 0:
                osa.auth_token_expires = soon
            osa.authenticate(force=False)
        self.assertEqual(mocked_auth_method.call_count, 1)

    def test_authentication_cache(self):
        tuples = [('1.1', OpenStackMockHttp, {}), ('2.0', OpenStack_2_0_MockHttp, {}), ('2.0_apikey', OpenStack_2_0_MockHttp, {}), ('2.0_password', OpenStack_2_0_MockHttp, {}), ('3.x_password', OpenStackIdentity_3_0_MockHttp, {'user_id': 'test_user_id', 'key': 'test_key', 'token_scope': 'project', 'tenant_name': 'test_tenant', 'tenant_domain_id': 'test_tenant_domain_id', 'domain_name': 'test_domain'}), ('3.x_oidc_access_token', OpenStackIdentity_3_0_MockHttp, {'user_id': 'test_user_id', 'key': 'test_key', 'token_scope': 'domain', 'tenant_name': 'test_tenant', 'tenant_domain_id': 'test_tenant_domain_id', 'domain_name': 'test_domain'})]
        user_id = OPENSTACK_PARAMS[0]
        key = OPENSTACK_PARAMS[1]
        for auth_version, mock_http_class, kwargs in tuples:
            mock_http_class.type = None
            connection = self._get_mock_connection(mock_http_class=mock_http_class)
            auth_url = connection.auth_url
            if not kwargs:
                kwargs['user_id'] = user_id
                kwargs['key'] = key
            auth_cache = OpenStackMockAuthCache()
            self.assertEqual(len(auth_cache), 0)
            kwargs['auth_cache'] = auth_cache
            cls = get_class_for_auth_version(auth_version=auth_version)
            osa = cls(auth_url=auth_url, parent_conn=connection, **kwargs)
            osa = osa.authenticate()
            self.assertEqual(len(auth_cache), 1)
            osa = cls(auth_url=auth_url, parent_conn=connection, **kwargs)
            osa.request = Mock(wraps=osa.request)
            osa = osa.authenticate()
            if auth_version in ('1.1', '2.0', '2.0_apikey', '2.0_password'):
                self.assertEqual(osa.request.call_count, 0)
            elif auth_version in ('3.x_password', '3.x_oidc_access_token'):
                osa.request.assert_called_once_with(action='/v3/auth/tokens', params=None, data=None, headers={'X-Subject-Token': '00000000000000000000000000000000', 'X-Auth-Token': '00000000000000000000000000000000'}, method='GET', raw=False)
            self.assertEqual(len(auth_cache), 1)
            cache_key = list(auth_cache.store.keys())[0]
            auth_context = auth_cache.get(cache_key)
            auth_context.expiration = YESTERDAY
            auth_cache.put(cache_key, auth_context)
            osa = cls(auth_url=auth_url, parent_conn=connection, **kwargs)
            osa.request = Mock(wraps=osa.request)
            osa._get_unscoped_token_from_oidc_token = Mock(return_value='000')
            OpenStackIdentity_3_0_MockHttp.type = 'GET_UNAUTHORIZED_POST_OK'
            osa = osa.authenticate()
            if auth_version in ('1.1', '2.0', '2.0_apikey', '2.0_password'):
                self.assertEqual(osa.request.call_count, 1)
                self.assertTrue(osa.request.call_args[1]['method'], 'POST')
            elif auth_version in ('3.x_password', '3.x_oidc_access_token'):
                self.assertTrue(osa.request.call_args[0][0], '/v3/auth/tokens')
                self.assertTrue(osa.request.call_args[1]['method'], 'POST')
            if hasattr(osa, 'list_projects'):
                mock_http_class.type = None
                auth_cache.reset()
                osa = cls(auth_url=auth_url, parent_conn=connection, **kwargs)
                osa.request = Mock(wraps=osa.request)
                osa = osa.authenticate()
                self.assertEqual(len(auth_cache), 1)
                mock_http_class.type = 'UNAUTHORIZED'
                try:
                    osa.list_projects()
                except:
                    pass
                self.assertEqual(len(auth_cache), 0)

    def _get_mock_connection(self, mock_http_class, auth_url=None):
        OpenStackBaseConnection.conn_class = mock_http_class
        if auth_url is None:
            auth_url = 'https://auth.api.example.com'
        OpenStackBaseConnection.auth_url = auth_url
        connection = OpenStackBaseConnection(*OPENSTACK_PARAMS)
        connection._ex_force_base_url = 'https://www.foo.com'
        connection.driver = OpenStack_1_0_NodeDriver(*OPENSTACK_PARAMS)
        return connection