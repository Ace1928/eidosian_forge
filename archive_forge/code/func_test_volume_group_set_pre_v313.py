from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group
def test_volume_group_set_pre_v313(self):
    self.volume_client.api_version = api_versions.APIVersion('3.12')
    arglist = [self.fake_volume_group.id, '--name', 'foo', '--description', 'hello, world']
    verifylist = [('group', self.fake_volume_group.id), ('name', 'foo'), ('description', 'hello, world')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-volume-api-version 3.13 or greater is required', str(exc))