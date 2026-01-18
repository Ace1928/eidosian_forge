from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v2 import type
from openstack.tests.unit import base
def test_remove_private_access(self):
    sot = type.Type(**TYPE)
    self.assertIsNone(sot.remove_private_access(self.sess, 'a'))
    url = 'types/%s/action' % sot.id
    body = {'removeProjectAccess': {'project': 'a'}}
    self.sess.post.assert_called_with(url, json=body)