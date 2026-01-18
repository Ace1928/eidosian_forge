from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
def test_obj_to_munch(self):
    cloud = FakeCloud()
    cloud.subcloud = FakeCloud()
    cloud_dict = meta.obj_to_munch(cloud)
    self.assertEqual(FakeCloud.name, cloud_dict['name'])
    self.assertNotIn('_unused', cloud_dict)
    self.assertNotIn('get_flavor_name', cloud_dict)
    self.assertNotIn('subcloud', cloud_dict)
    self.assertTrue(hasattr(cloud_dict, 'name'))
    self.assertEqual(cloud_dict.name, cloud_dict['name'])