import ddt
from tempest.lib import exceptions
from cinderclient.tests.functional import base
@ddt.ddt
class CinderVolumeExtendNegativeTests(base.ClientTestBase):
    """Check of cinder volume extend command."""

    def setUp(self):
        super(CinderVolumeExtendNegativeTests, self).setUp()
        self.volume = self.object_create('volume', params='1')

    @ddt.data(('', 'too few arguments|the following arguments are required'), ('-1', 'Invalid input for field/attribute new_size. Value: -1. -1 is less than the minimum of 1'), ('0', 'Invalid input for field/attribute new_size. Value: 0. 0 is less than the minimum of 1'), ('size', 'invalid int value'), ('0.2', 'invalid int value'), ('2 GB', 'unrecognized arguments'), ('999999999', 'VolumeSizeExceedsAvailableQuota'))
    @ddt.unpack
    def test_volume_extend_with_incorrect_size(self, value, ex_text):
        self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.cinder, 'extend', params='{0} {1}'.format(self.volume['id'], value))

    @ddt.data(('', 'too few arguments|the following arguments are required'), ('1234-1234-1234', 'No volume with a name or ID of'), ('my_volume', 'No volume with a name or ID of'), ('1234 1234', 'unrecognized arguments'))
    @ddt.unpack
    def test_volume_extend_with_incorrect_volume_id(self, value, ex_text):
        self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.cinder, 'extend', params='{0} 2'.format(value))