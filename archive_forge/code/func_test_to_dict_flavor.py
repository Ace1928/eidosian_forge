from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_to_dict_flavor(self):
    sot = server.Server(**EXAMPLE)
    dct = sot.to_dict()
    self.assertEqual(0, dct['flavor']['vcpus'])