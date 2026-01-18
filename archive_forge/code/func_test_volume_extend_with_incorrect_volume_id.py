import ddt
from tempest.lib import exceptions
from cinderclient.tests.functional import base
@ddt.data(('', 'too few arguments|the following arguments are required'), ('1234-1234-1234', 'No volume with a name or ID of'), ('my_volume', 'No volume with a name or ID of'), ('1234 1234', 'unrecognized arguments'))
@ddt.unpack
def test_volume_extend_with_incorrect_volume_id(self, value, ex_text):
    self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.cinder, 'extend', params='{0} 2'.format(value))