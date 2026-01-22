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
class ManagerRequestIdTest(utils.TestCase):
    url = '/test-url'
    resp = create_response_with_request_id_header()

    def setUp(self):
        super(ManagerRequestIdTest, self).setUp()
        auth = v2.Token(auth_url='http://127.0.0.1:5000', token=self.TEST_TOKEN)
        session_ = session.Session(auth=auth)
        self.client = client.Client(session=session_, include_metadata='True')._adapter
        self.mgr = base.Manager(self.client)
        self.mgr.resource_class = base.Resource

    def mock_request_method(self, request_method, body):
        return self.useFixture(fixtures.MockPatchObject(self.client, request_method, autospec=True, return_value=(self.resp, body))).mock

    def test_get(self):
        body = {'hello': {'hi': 1}}
        get_mock = self.mock_request_method('get', body)
        rsrc = self.mgr._get(self.url, 'hello')
        get_mock.assert_called_once_with(self.url)
        self.assertEqual(rsrc.data.hi, 1)
        self.assertEqual(rsrc.request_ids[0], TEST_REQUEST_ID)

    def test_list(self):
        body = {'hello': [{'name': 'admin'}, {'name': 'admin'}]}
        get_mock = self.mock_request_method('get', body)
        returned_list = self.mgr._list(self.url, 'hello')
        self.assertEqual(returned_list.request_ids[0], TEST_REQUEST_ID)
        get_mock.assert_called_once_with(self.url)

    def test_list_with_multiple_response_objects(self):
        body = {'hello': [{'name': 'admin'}, {'name': 'admin'}]}
        resp_1 = requests.Response()
        resp_1.headers['x-openstack-request-id'] = TEST_REQUEST_ID
        resp_2 = requests.Response()
        resp_2.headers['x-openstack-request-id'] = TEST_REQUEST_ID_1
        resp_result = [resp_1, resp_2]
        get_mock = self.useFixture(fixtures.MockPatchObject(self.client, 'get', autospec=True, return_value=(resp_result, body))).mock
        returned_list = self.mgr._list(self.url, 'hello')
        self.assertIn(returned_list.request_ids[0], [TEST_REQUEST_ID, TEST_REQUEST_ID_1])
        self.assertIn(returned_list.request_ids[1], [TEST_REQUEST_ID, TEST_REQUEST_ID_1])
        get_mock.assert_called_once_with(self.url)

    def test_post(self):
        body = {'hello': {'hi': 1}}
        post_mock = self.mock_request_method('post', body)
        rsrc = self.mgr._post(self.url, body, 'hello')
        post_mock.assert_called_once_with(self.url, body=body)
        self.assertEqual(rsrc.data.hi, 1)
        post_mock.reset_mock()
        rsrc = self.mgr._post(self.url, body, 'hello', return_raw=True)
        post_mock.assert_called_once_with(self.url, body=body)
        self.assertNotIsInstance(rsrc, base.Response)
        self.assertEqual(rsrc['hi'], 1)

    def test_put(self):
        body = {'hello': {'hi': 1}}
        put_mock = self.mock_request_method('put', body)
        rsrc = self.mgr._put(self.url, body, 'hello')
        put_mock.assert_called_once_with(self.url, body=body)
        self.assertEqual(rsrc.data.hi, 1)
        put_mock.reset_mock()
        rsrc = self.mgr._put(self.url, body)
        put_mock.assert_called_once_with(self.url, body=body)
        self.assertEqual(rsrc.data.hello['hi'], 1)
        self.assertEqual(rsrc.request_ids[0], TEST_REQUEST_ID)

    def test_head(self):
        get_mock = self.mock_request_method('head', None)
        rsrc = self.mgr._head(self.url)
        get_mock.assert_called_once_with(self.url)
        self.assertFalse(rsrc.data)
        self.assertEqual(rsrc.request_ids[0], TEST_REQUEST_ID)

    def test_delete(self):
        delete_mock = self.mock_request_method('delete', None)
        resp, base_resp = self.mgr._delete(self.url, name='hello')
        delete_mock.assert_called_once_with('/test-url', name='hello')
        self.assertEqual(base_resp.request_ids[0], TEST_REQUEST_ID)
        self.assertEqual(base_resp.data, None)
        self.assertIsInstance(resp, requests.Response)

    def test_patch(self):
        body = {'hello': {'hi': 1}}
        patch_mock = self.mock_request_method('patch', body)
        rsrc = self.mgr._patch(self.url, body, 'hello')
        patch_mock.assert_called_once_with(self.url, body=body)
        self.assertEqual(rsrc.data.hi, 1)
        patch_mock.reset_mock()
        rsrc = self.mgr._patch(self.url, body)
        patch_mock.assert_called_once_with(self.url, body=body)
        self.assertEqual(rsrc.data.hello['hi'], 1)
        self.assertEqual(rsrc.request_ids[0], TEST_REQUEST_ID)

    def test_update(self):
        body = {'hello': {'hi': 1}}
        patch_mock = self.mock_request_method('patch', body)
        put_mock = self.mock_request_method('put', body)
        rsrc = self.mgr._update(self.url, body=body, response_key='hello', method='PATCH', management=False)
        patch_mock.assert_called_once_with(self.url, management=False, body=body)
        self.assertEqual(rsrc.data.hi, 1)
        rsrc = self.mgr._update(self.url, body=None, response_key='hello', method='PUT', management=True)
        put_mock.assert_called_once_with(self.url, management=True, body=None)
        self.assertEqual(rsrc.data.hi, 1)
        self.assertEqual(rsrc.request_ids[0], TEST_REQUEST_ID)