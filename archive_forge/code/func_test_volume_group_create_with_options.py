from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group
def test_volume_group_create_with_options(self):
    self.volume_client.api_version = api_versions.APIVersion('3.13')
    arglist = ['--volume-group-type', self.fake_volume_group_type.id, '--volume-type', self.fake_volume_type.id, '--name', 'foo', '--description', 'hello, world', '--availability-zone', 'bar']
    verifylist = [('volume_group_type', self.fake_volume_group_type.id), ('volume_types', [self.fake_volume_type.id]), ('name', 'foo'), ('description', 'hello, world'), ('availability_zone', 'bar')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_group_types_mock.get.assert_called_once_with(self.fake_volume_group_type.id)
    self.volume_types_mock.get.assert_called_once_with(self.fake_volume_type.id)
    self.volume_groups_mock.create.assert_called_once_with(self.fake_volume_group_type.id, self.fake_volume_type.id, 'foo', 'hello, world', availability_zone='bar')
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)