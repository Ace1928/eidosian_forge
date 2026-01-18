from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_reset_state(self):
    sot = server.Server(**EXAMPLE)
    self.assertIsNone(sot.reset_state(self.sess, 'active'))
    url = 'servers/IDENTIFIER/action'
    body = {'os-resetState': {'state': 'active'}}
    headers = {'Accept': ''}
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=self.sess.default_microversion)