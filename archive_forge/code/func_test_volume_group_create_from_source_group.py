from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group
def test_volume_group_create_from_source_group(self):
    self.volume_client.api_version = api_versions.APIVersion('3.14')
    arglist = ['--source-group', self.fake_volume_group.id]
    verifylist = [('source_group', self.fake_volume_group.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_groups_mock.get.assert_has_calls([mock.call(self.fake_volume_group.id), mock.call(self.fake_volume_group.id)])
    self.volume_groups_mock.create_from_src.assert_called_once_with(None, self.fake_volume_group.id, None, None)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)