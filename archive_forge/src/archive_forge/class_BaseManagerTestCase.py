from unittest import mock
from oslotest import base as test_base
from ironicclient.common.apiclient import base
class BaseManagerTestCase(test_base.BaseTestCase):

    def setUp(self):
        super(BaseManagerTestCase, self).setUp()
        self.response = mock.MagicMock()
        self.http_client = mock.MagicMock()
        self.http_client.get.return_value = self.response
        self.http_client.post.return_value = self.response
        self.manager = base.BaseManager(self.http_client)
        self.manager.resource_class = HumanResource

    def test_list(self):
        self.response.json.return_value = {'human_resources': [{'id': 42}]}
        expected = [HumanResource(self.manager, {'id': 42}, loaded=True)]
        result = self.manager._list('/human_resources', 'human_resources')
        self.assertEqual(expected, result)

    def test_list_no_response_key(self):
        self.response.json.return_value = [{'id': 42}]
        expected = [HumanResource(self.manager, {'id': 42}, loaded=True)]
        result = self.manager._list('/human_resources')
        self.assertEqual(expected, result)

    def test_list_get(self):
        self.manager._list('/human_resources', 'human_resources')
        self.manager.client.get.assert_called_with('/human_resources')

    def test_list_post(self):
        self.manager._list('/human_resources', 'human_resources', json={'id': 42})
        self.manager.client.post.assert_called_with('/human_resources', json={'id': 42})

    def test_get(self):
        self.response.json.return_value = {'human_resources': {'id': 42}}
        expected = HumanResource(self.manager, {'id': 42}, loaded=True)
        result = self.manager._get('/human_resources/42', 'human_resources')
        self.manager.client.get.assert_called_with('/human_resources/42')
        self.assertEqual(expected, result)

    def test_get_no_response_key(self):
        self.response.json.return_value = {'id': 42}
        expected = HumanResource(self.manager, {'id': 42}, loaded=True)
        result = self.manager._get('/human_resources/42')
        self.manager.client.get.assert_called_with('/human_resources/42')
        self.assertEqual(expected, result)

    def test_post(self):
        self.response.json.return_value = {'human_resources': {'id': 42}}
        expected = HumanResource(self.manager, {'id': 42}, loaded=True)
        result = self.manager._post('/human_resources', response_key='human_resources', json={'id': 42})
        self.manager.client.post.assert_called_with('/human_resources', json={'id': 42})
        self.assertEqual(expected, result)

    def test_post_return_raw(self):
        self.response.json.return_value = {'human_resources': {'id': 42}}
        result = self.manager._post('/human_resources', response_key='human_resources', json={'id': 42}, return_raw=True)
        self.manager.client.post.assert_called_with('/human_resources', json={'id': 42})
        self.assertEqual({'id': 42}, result)

    def test_post_no_response_key(self):
        self.response.json.return_value = {'id': 42}
        expected = HumanResource(self.manager, {'id': 42}, loaded=True)
        result = self.manager._post('/human_resources', json={'id': 42})
        self.manager.client.post.assert_called_with('/human_resources', json={'id': 42})
        self.assertEqual(expected, result)