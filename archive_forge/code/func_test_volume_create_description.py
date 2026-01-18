import ddt
from tempest.lib import exceptions
from cinderclient.tests.functional import base
def test_volume_create_description(self):
    """Test steps:

        1) create volume with description
        2) check that volume has right description
        """
    volume_description = 'test_description'
    volume = self.object_create('volume', params='--description {0} 1'.format(volume_description))
    self.assertEqual(volume_description, volume['description'])