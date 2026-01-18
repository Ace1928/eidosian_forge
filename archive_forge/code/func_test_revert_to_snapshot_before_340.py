import copy
from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import volume
from openstack import exceptions
from openstack.tests.unit import base
@mock.patch('openstack.utils.require_microversion', autospec=True, side_effect=[exceptions.SDKException()])
def test_revert_to_snapshot_before_340(self, mv_mock):
    sot = volume.Volume(**VOLUME)
    self.assertRaises(exceptions.SDKException, sot.revert_to_snapshot, self.sess, '1')