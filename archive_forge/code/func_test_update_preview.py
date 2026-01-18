from unittest import mock
from openstack import exceptions
from openstack.orchestration.v1 import stack
from openstack import resource
from openstack.tests.unit import base
from openstack.tests.unit import test_resource
def test_update_preview(self):
    sess = mock.Mock()
    sess.default_microversion = None
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.json.return_value = FAKE_UPDATE_PREVIEW_RESPONSE.copy()
    sess.put = mock.Mock(return_value=mock_response)
    sot = stack.Stack(**FAKE)
    body = sot._body.dirty.copy()
    ret = sot.update(sess, preview=True)
    sess.put.assert_called_with('stacks/%s/%s/preview' % (FAKE_NAME, FAKE_ID), headers={}, microversion=None, json=body)
    self.assertEqual(FAKE_UPDATE_PREVIEW_RESPONSE['added'], ret.added)
    self.assertEqual(FAKE_UPDATE_PREVIEW_RESPONSE['deleted'], ret.deleted)
    self.assertEqual(FAKE_UPDATE_PREVIEW_RESPONSE['replaced'], ret.replaced)
    self.assertEqual(FAKE_UPDATE_PREVIEW_RESPONSE['unchanged'], ret.unchanged)
    self.assertEqual(FAKE_UPDATE_PREVIEW_RESPONSE['updated'], ret.updated)