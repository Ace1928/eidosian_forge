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
class ProtocolRequestIdTests(utils.TestRequestId):

    def setUp(self):
        super(ProtocolRequestIdTests, self).setUp()
        self.mgr = protocols.ProtocolManager(self.client)

    def _mock_request_method(self, method=None, body=None):
        return self.useFixture(fixtures.MockPatchObject(self.client, method, autospec=True, return_value=(self.resp, body))).mock

    def test_get_protocol(self):
        body = {'protocol': {'name': 'admin'}}
        get_mock = self._mock_request_method(method='get', body=body)
        response = self.mgr.get('admin', 'protocol')
        self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
        get_mock.assert_called_once_with('OS-FEDERATION/identity_providers/admin/protocols/protocol')

    def test_list_protocol(self):
        body = {'protocols': [{'name': 'admin'}]}
        get_mock = self._mock_request_method(method='get', body=body)
        response = self.mgr.list('identity_provider')
        self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
        get_mock.assert_called_once_with('OS-FEDERATION/identity_providers/identity_provider/protocols?')

    def test_create_protocol(self):
        body = {'protocol': {'name': 'admin'}}
        self._mock_request_method(method='post', body=body)
        put_mock = self._mock_request_method(method='put', body=body)
        response = self.mgr.create(protocol_id='admin', identity_provider='fake', mapping='fake')
        self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
        put_mock.assert_called_once_with('OS-FEDERATION/identity_providers/fake/protocols/admin', body={'protocol': {'mapping_id': 'fake'}})

    def test_update_protocol(self):
        body = {'protocol': {'name': 'admin'}}
        patch_mock = self._mock_request_method(method='patch', body=body)
        self._mock_request_method(method='post', body=body)
        response = self.mgr.update(protocol='admin', identity_provider='fake', mapping='fake')
        self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
        patch_mock.assert_called_once_with('OS-FEDERATION/identity_providers/fake/protocols/admin', body={'protocol': {'mapping_id': 'fake'}})

    def test_delete_protocol(self):
        get_mock = self._mock_request_method(method='delete')
        _, resp = self.mgr.delete('identity_provider', 'protocol')
        self.assertEqual(resp.request_ids[0], self.TEST_REQUEST_ID)
        get_mock.assert_called_once_with('OS-FEDERATION/identity_providers/identity_provider/protocols/protocol')