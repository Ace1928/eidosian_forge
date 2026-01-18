from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v2 import backup
from openstack import exceptions
from openstack.tests.unit import base
def test_restore_name(self):
    sot = backup.Backup(**BACKUP)
    self.assertEqual(sot, sot.restore(self.sess, name='name'))
    url = 'backups/%s/restore' % FAKE_ID
    body = {'restore': {'name': 'name'}}
    self.sess.post.assert_called_with(url, json=body)