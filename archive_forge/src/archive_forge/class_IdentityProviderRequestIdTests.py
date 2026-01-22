import copy
import fixtures
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneauth1.tests.unit import k2k_fixtures
from testtools import matchers
from keystoneclient import access
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import client
from keystoneclient.v3.contrib.federation import base
from keystoneclient.v3.contrib.federation import identity_providers
from keystoneclient.v3.contrib.federation import mappings
from keystoneclient.v3.contrib.federation import protocols
from keystoneclient.v3.contrib.federation import service_providers
from keystoneclient.v3 import domains
from keystoneclient.v3 import projects
class IdentityProviderRequestIdTests(utils.TestRequestId):

    def setUp(self):
        super(IdentityProviderRequestIdTests, self).setUp()
        self.mgr = identity_providers.IdentityProviderManager(self.client)

    def _mock_request_method(self, method=None, body=None):
        return self.useFixture(fixtures.MockPatchObject(self.client, method, autospec=True, return_value=(self.resp, body))).mock

    def test_get_identity_provider(self):
        body = {'identity_provider': {'name': 'admin'}}
        get_mock = self._mock_request_method(method='get', body=body)
        response = self.mgr.get('admin')
        self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
        get_mock.assert_called_once_with('OS-FEDERATION/identity_providers/admin')

    def test_list_identity_provider(self):
        body = {'identity_providers': [{'name': 'admin'}]}
        get_mock = self._mock_request_method(method='get', body=body)
        response = self.mgr.list()
        self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
        get_mock.assert_called_once_with('OS-FEDERATION/identity_providers?')

    def test_create_identity_provider(self):
        body = {'identity_provider': {'name': 'admin'}}
        self._mock_request_method(method='post', body=body)
        put_mock = self._mock_request_method(method='put', body=body)
        response = self.mgr.create(id='admin', description='fake')
        self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
        put_mock.assert_called_once_with('OS-FEDERATION/identity_providers/admin', body={'identity_provider': {'description': 'fake'}})

    def test_update_identity_provider(self):
        body = {'identity_provider': {'name': 'admin'}}
        patch_mock = self._mock_request_method(method='patch', body=body)
        self._mock_request_method(method='post', body=body)
        response = self.mgr.update('admin')
        self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
        patch_mock.assert_called_once_with('OS-FEDERATION/identity_providers/admin', body={'identity_provider': {}})

    def test_delete_identity_provider(self):
        get_mock = self._mock_request_method(method='delete')
        _, resp = self.mgr.delete('admin')
        self.assertEqual(resp.request_ids[0], self.TEST_REQUEST_ID)
        get_mock.assert_called_once_with('OS-FEDERATION/identity_providers/admin')