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
def test_type_show_with_encryption(self):
    encryption_type = volume_fakes.create_one_encryption_volume_type()
    encryption_info = {'provider': 'LuksEncryptor', 'cipher': None, 'key_size': None, 'control_location': 'front-end'}
    self.volume_type = volume_fakes.create_one_volume_type(attrs={'encryption': encryption_info})
    self.volume_types_mock.get.return_value = self.volume_type
    self.volume_encryption_types_mock.get.return_value = encryption_type
    encryption_columns = ('access_project_ids', 'description', 'encryption', 'id', 'is_public', 'name', 'properties')
    encryption_data = (None, self.volume_type.description, format_columns.DictColumn(encryption_info), self.volume_type.id, True, self.volume_type.name, format_columns.DictColumn(self.volume_type.extra_specs))
    arglist = ['--encryption-type', self.volume_type.id]
    verifylist = [('encryption_type', True), ('volume_type', self.volume_type.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_types_mock.get.assert_called_with(self.volume_type.id)
    self.volume_encryption_types_mock.get.assert_called_with(self.volume_type.id)
    self.assertEqual(encryption_columns, columns)
    self.assertCountEqual(encryption_data, data)