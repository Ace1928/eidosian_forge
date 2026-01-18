from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_rescue_with_options(self):
    sot = server.Server(**EXAMPLE)
    res = sot.rescue(self.sess, admin_pass='SECRET', image_ref='IMG-ID')
    self.assertIsNone(res)
    url = 'servers/IDENTIFIER/action'
    body = {'rescue': {'adminPass': 'SECRET', 'rescue_image_ref': 'IMG-ID'}}
    headers = {'Accept': ''}
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=self.sess.default_microversion)