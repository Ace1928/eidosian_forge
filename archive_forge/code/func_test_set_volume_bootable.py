from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v2 import volume
from openstack.tests.unit import base
def test_set_volume_bootable(self):
    sot = volume.Volume(**VOLUME)
    self.assertIsNone(sot.set_bootable_status(self.sess))
    url = 'volumes/%s/action' % FAKE_ID
    body = {'os-set_bootable': {'bootable': True}}
    self.sess.post.assert_called_with(url, json=body, microversion=sot._max_microversion)