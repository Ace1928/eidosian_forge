from unittest import mock
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group_type
def test_volume_group_type_set(self):
    self.volume_client.api_version = api_versions.APIVersion('3.11')
    self.fake_volume_group_type.set_keys.return_value = None
    arglist = [self.fake_volume_group_type.id, '--name', 'foo', '--description', 'hello, world', '--public', '--property', 'fizz=buzz']
    verifylist = [('group_type', self.fake_volume_group_type.id), ('name', 'foo'), ('description', 'hello, world'), ('is_public', True), ('no_property', False), ('properties', {'fizz': 'buzz'})]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_group_types_mock.update.assert_called_once_with(self.fake_volume_group_type.id, name='foo', description='hello, world', is_public=True)
    self.fake_volume_group_type.set_keys.assert_called_once_with({'fizz': 'buzz'})
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)