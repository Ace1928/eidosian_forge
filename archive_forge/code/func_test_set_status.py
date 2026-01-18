import copy
from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import snapshot
from openstack.tests.unit import base
def test_set_status(self):
    sot = snapshot.Snapshot(**SNAPSHOT)
    self.assertIsNone(sot.set_status(self.sess, 'new_status'))
    url = 'snapshots/%s/action' % FAKE_ID
    body = {'os-update_snapshot_status': {'status': 'new_status'}}
    self.sess.post.assert_called_with(url, json=body, microversion=sot._max_microversion)