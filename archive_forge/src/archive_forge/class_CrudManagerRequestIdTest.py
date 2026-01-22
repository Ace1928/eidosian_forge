import uuid
import fixtures
from keystoneauth1.identity import v2
from keystoneauth1 import session
import requests
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.tests.unit import utils
from keystoneclient import utils as base_utils
from keystoneclient.v2_0 import client
from keystoneclient.v2_0 import roles
from keystoneclient.v3 import users
class CrudManagerRequestIdTest(utils.TestCase):
    resp = create_response_with_request_id_header()
    request_resp = requests.Response()
    request_resp.headers['x-openstack-request-id'] = TEST_REQUEST_ID

    def setUp(self):
        super(CrudManagerRequestIdTest, self).setUp()
        auth = v2.Token(auth_url='http://127.0.0.1:5000', token=self.TEST_TOKEN)
        session_ = session.Session(auth=auth)
        self.client = client.Client(session=session_, include_metadata='True')._adapter

    def test_find_resource(self):
        body = {'users': [{'name': 'entity_one'}]}
        get_mock = self.useFixture(fixtures.MockPatchObject(self.client, 'get', autospec=True, side_effect=[exceptions.NotFound, (self.request_resp, body)])).mock
        mgr = users.UserManager(self.client)
        mgr.resource_class = users.User
        response = base_utils.find_resource(mgr, 'entity_one')
        get_mock.assert_called_with('/users?name=entity_one')
        self.assertEqual(response.request_ids[0], TEST_REQUEST_ID)

    def test_list(self):
        body = {'users': [{'name': 'admin'}, {'name': 'admin'}]}
        get_mock = self.useFixture(fixtures.MockPatchObject(self.client, 'get', autospec=True, return_value=(self.request_resp, body))).mock
        mgr = users.UserManager(self.client)
        mgr.resource_class = users.User
        returned_list = mgr.list()
        self.assertEqual(returned_list.request_ids[0], TEST_REQUEST_ID)
        get_mock.assert_called_once_with('/users?')