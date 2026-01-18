import ddt
from tempest.lib import exceptions
from cinderclient.tests.functional import base
def test_volume_create_metadata(self):
    """Test steps:

        1) create volume with metadata
        2) check that metadata complies entered
        """
    volume = self.object_create('volume', params='--metadata test_metadata=test_date 1')
    self.assertEqual(str({'test_metadata': 'test_date'}), volume['metadata'])