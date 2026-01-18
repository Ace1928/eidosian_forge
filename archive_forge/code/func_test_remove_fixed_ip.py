from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_remove_fixed_ip(self):
    sot = server.Server(**EXAMPLE)
    res = sot.remove_fixed_ip(self.sess, 'ADDRESS')
    self.assertIsNone(res)
    url = 'servers/IDENTIFIER/action'
    body = {'removeFixedIp': {'address': 'ADDRESS'}}
    headers = {'Accept': ''}
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=self.sess.default_microversion)