from unittest import mock
from keystoneauth1 import session
from requests_mock.contrib import fixture
from openstackclient.api import object_store_v1 as object_store
from openstackclient.tests.unit import utils
def test_object_list_marker_limit_end(self):
    self.requests_mock.register_uri('GET', FAKE_URL + '/qaz?marker=next&limit=2&end_marker=stop', json=LIST_CONTAINER_RESP, status_code=200)
    ret = self.api.object_list(container='qaz', marker='next', limit=2, end_marker='stop')
    self.assertEqual(LIST_CONTAINER_RESP, ret)