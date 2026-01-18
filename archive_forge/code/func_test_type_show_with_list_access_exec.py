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
def test_type_show_with_list_access_exec(self):
    arglist = [self.volume_type.id]
    verifylist = [('volume_type', self.volume_type.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    private_type = volume_fakes.create_one_volume_type(attrs={'is_public': False})
    with mock.patch.object(self.volume_types_mock, 'get', return_value=private_type):
        with mock.patch.object(self.volume_type_access_mock, 'list', side_effect=Exception()):
            columns, data = self.cmd.take_action(parsed_args)
            self.volume_types_mock.get.assert_called_once_with(self.volume_type.id)
            self.volume_type_access_mock.list.assert_called_once_with(private_type.id)
    self.assertEqual(self.columns, columns)
    private_type_data = (None, private_type.description, private_type.id, private_type.is_public, private_type.name, format_columns.DictColumn(private_type.extra_specs))
    self.assertCountEqual(private_type_data, data)