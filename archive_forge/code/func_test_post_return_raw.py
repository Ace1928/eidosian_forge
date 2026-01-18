from unittest import mock
from oslotest import base as test_base
from ironicclient.common.apiclient import base
def test_post_return_raw(self):
    self.response.json.return_value = {'human_resources': {'id': 42}}
    result = self.manager._post('/human_resources', response_key='human_resources', json={'id': 42}, return_raw=True)
    self.manager.client.post.assert_called_with('/human_resources', json={'id': 42})
    self.assertEqual({'id': 42}, result)