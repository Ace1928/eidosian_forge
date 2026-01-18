from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group
def test_volume_group_failover_pre_v338(self):
    self.volume_client.api_version = api_versions.APIVersion('3.37')
    arglist = [self.fake_volume_group.id, '--allow-attached-volume', '--secondary-backend-id', 'foo']
    verifylist = [('group', self.fake_volume_group.id), ('allow_attached_volume', True), ('secondary_backend_id', 'foo')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-volume-api-version 3.38 or greater is required', str(exc))