import copy
from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import volume
from openstack import exceptions
from openstack.tests.unit import base
def test_set_volume_readonly_false(self):
    sot = volume.Volume(**VOLUME)
    self.assertIsNone(sot.set_readonly(self.sess, False))
    url = 'volumes/%s/action' % FAKE_ID
    body = {'os-update_readonly_flag': {'readonly': False}}
    self.sess.post.assert_called_with(url, json=body, microversion=sot._max_microversion)