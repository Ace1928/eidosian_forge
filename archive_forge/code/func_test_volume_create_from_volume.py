import ddt
from tempest.lib import exceptions
from cinderclient.tests.functional import base
def test_volume_create_from_volume(self):
    """Test steps:

        1) create volume in Setup()
        2) create volume from volume
        3) check that volume from volume has been successfully created
        """
    volume_from_volume = self.object_create('volume', params='--source-volid {0} 1'.format(self.volume['id']))
    cinder_list = self.cinder('list')
    self.assertIn(volume_from_volume['id'], cinder_list)