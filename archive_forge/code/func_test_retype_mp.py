from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v2 import volume
from openstack.tests.unit import base
def test_retype_mp(self):
    sot = volume.Volume(**VOLUME)
    self.assertIsNone(sot.retype(self.sess, '1', migration_policy='2'))
    url = 'volumes/%s/action' % FAKE_ID
    body = {'os-retype': {'new_type': '1', 'migration_policy': '2'}}
    self.sess.post.assert_called_with(url, json=body, microversion=sot._max_microversion)