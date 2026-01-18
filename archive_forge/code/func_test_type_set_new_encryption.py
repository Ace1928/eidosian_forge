from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v1 import fakes as volume_fakes
from openstackclient.volume.v1 import volume_type
def test_type_set_new_encryption(self):
    arglist = ['--encryption-provider', 'LuksEncryptor', '--encryption-cipher', 'aes-xts-plain64', '--encryption-key-size', '128', '--encryption-control-location', 'front-end', self.volume_type.id]
    verifylist = [('encryption_provider', 'LuksEncryptor'), ('encryption_cipher', 'aes-xts-plain64'), ('encryption_key_size', 128), ('encryption_control_location', 'front-end'), ('volume_type', self.volume_type.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    body = {'provider': 'LuksEncryptor', 'cipher': 'aes-xts-plain64', 'key_size': 128, 'control_location': 'front-end'}
    self.encryption_types_mock.create.assert_called_with(self.volume_type, body)
    self.assertIsNone(result)