from fixtures import TimeoutException
from testtools import content
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_update_volume(self):
    name, desc = (self.getUniqueString('name'), self.getUniqueString('desc'))
    self.addCleanup(self.cleanup, name)
    volume = self.user_cloud.create_volume(1, name=name, description=desc)
    self.assertEqual(volume.name, name)
    self.assertEqual(volume.description, desc)
    new_name = self.getUniqueString('name')
    volume = self.user_cloud.update_volume(volume.id, name=new_name)
    self.assertNotEqual(volume.name, name)
    self.assertEqual(volume.name, new_name)
    self.assertEqual(volume.description, desc)