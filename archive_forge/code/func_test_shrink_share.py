from unittest import mock
from keystoneauth1 import adapter
from openstack.shared_file_system.v2 import share
from openstack.tests.unit import base
def test_shrink_share(self):
    sot = share.Share(**EXAMPLE)
    microversion = sot._get_microversion(self.sess, action='patch')
    self.assertIsNone(sot.shrink_share(self.sess, new_size=1))
    url = f'shares/{IDENTIFIER}/action'
    body = {'shrink': {'new_size': 1}}
    headers = {'Accept': ''}
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=microversion)