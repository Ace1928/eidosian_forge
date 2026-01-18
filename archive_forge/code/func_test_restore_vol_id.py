from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v2 import backup
from openstack import exceptions
from openstack.tests.unit import base
def test_restore_vol_id(self):
    sot = backup.Backup(**BACKUP)
    self.assertEqual(sot, sot.restore(self.sess, volume_id='vol'))
    url = 'backups/%s/restore' % FAKE_ID
    body = {'restore': {'volume_id': 'vol'}}
    self.sess.post.assert_called_with(url, json=body)