from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_attachment
def test_volume_attachment_create_with_mode_pre_v354(self):
    self.volume_client.api_version = api_versions.APIVersion('3.53')
    arglist = [self.volume.id, self.server.id, '--mode', 'rw']
    verifylist = [('volume', self.volume.id), ('server', self.server.id), ('mode', 'rw')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-volume-api-version 3.54 or greater is required', str(exc))