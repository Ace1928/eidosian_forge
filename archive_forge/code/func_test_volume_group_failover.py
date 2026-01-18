from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group
def test_volume_group_failover(self):
    self.volume_client.api_version = api_versions.APIVersion('3.38')
    arglist = [self.fake_volume_group.id, '--allow-attached-volume', '--secondary-backend-id', 'foo']
    verifylist = [('group', self.fake_volume_group.id), ('allow_attached_volume', True), ('secondary_backend_id', 'foo')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.volume_groups_mock.failover_replication.assert_called_once_with(self.fake_volume_group.id, allow_attached_volume=True, secondary_backend_id='foo')
    self.assertIsNone(result)