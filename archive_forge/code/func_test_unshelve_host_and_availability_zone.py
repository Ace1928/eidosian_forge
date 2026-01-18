from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_unshelve_host_and_availability_zone(self):
    sot = server.Server(**EXAMPLE)
    res = sot.unshelve(self.sess, availability_zone=sot.availability_zone, host=sot.hypervisor_hostname)
    self.assertIsNone(res)
    url = 'servers/IDENTIFIER/action'
    body = {'unshelve': {'availability_zone': sot.availability_zone, 'host': sot.hypervisor_hostname}}
    headers = {'Accept': ''}
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=self.sess.default_microversion)