import ddt
from tempest.lib import exceptions
from cinderclient.tests.functional import base
@ddt.data(('', 'Size is a required parameter'), ('-1', 'Invalid input for field/attribute size'), ('0', 'Invalid input for field/attribute size'), ('size', 'invalid int value'), ('0.2', 'invalid int value'), ('2 GB', 'unrecognized arguments'), ('999999999', 'VolumeSizeExceedsAvailableQuota'))
@ddt.unpack
def test_volume_create_with_incorrect_size(self, value, ex_text):
    self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.object_create, 'volume', params=value)