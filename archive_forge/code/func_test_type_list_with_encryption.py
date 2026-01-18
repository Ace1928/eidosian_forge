from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_type
def test_type_list_with_encryption(self):
    encryption_type = volume_fakes.create_one_encryption_volume_type(attrs={'volume_type_id': self.volume_types[0].id})
    encryption_info = {'provider': 'LuksEncryptor', 'cipher': None, 'key_size': None, 'control_location': 'front-end'}
    encryption_columns = self.columns + ['Encryption']
    encryption_data = []
    encryption_data.append((self.volume_types[0].id, self.volume_types[0].name, self.volume_types[0].is_public, volume_type.EncryptionInfoColumn(self.volume_types[0].id, {self.volume_types[0].id: encryption_info})))
    encryption_data.append((self.volume_types[1].id, self.volume_types[1].name, self.volume_types[1].is_public, volume_type.EncryptionInfoColumn(self.volume_types[1].id, {})))
    self.volume_encryption_types_mock.list.return_value = [encryption_type]
    arglist = ['--encryption-type']
    verifylist = [('encryption_type', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_encryption_types_mock.list.assert_called_once_with()
    self.volume_types_mock.list.assert_called_once_with(search_opts={}, is_public=None)
    self.assertEqual(encryption_columns, columns)
    self.assertCountEqual(encryption_data, list(data))