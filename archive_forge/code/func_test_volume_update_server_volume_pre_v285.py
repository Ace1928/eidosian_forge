from unittest import mock
from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import volumes
def test_volume_update_server_volume_pre_v285(self):
    self.cs.api_version = api_versions.APIVersion('2.84')
    ex = self.assertRaises(TypeError, self.cs.volumes.update_server_volume, '1234', 'Work', 'Work', delete_on_termination=True)
    self.assertIn('delete_on_termination', str(ex))