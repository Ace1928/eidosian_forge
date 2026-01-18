from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_remove_security_group(self):
    sot = server.Server(**EXAMPLE)
    self.assertIsNone(sot.remove_security_group(self.sess, 'group'))
    url = 'servers/IDENTIFIER/action'
    body = {'removeSecurityGroup': {'name': 'group'}}
    headers = {'Accept': ''}
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=self.sess.default_microversion)