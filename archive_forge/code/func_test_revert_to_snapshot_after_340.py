import copy
from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import volume
from openstack import exceptions
from openstack.tests.unit import base
@mock.patch('openstack.utils.require_microversion', autospec=True, side_effect=[None])
def test_revert_to_snapshot_after_340(self, mv_mock):
    sot = volume.Volume(**VOLUME)
    self.assertIsNone(sot.revert_to_snapshot(self.sess, '1'))
    url = 'volumes/%s/action' % FAKE_ID
    body = {'revert': {'snapshot_id': '1'}}
    self.sess.post.assert_called_with(url, json=body, microversion=sot._max_microversion)
    mv_mock.assert_called_with(self.sess, '3.40')