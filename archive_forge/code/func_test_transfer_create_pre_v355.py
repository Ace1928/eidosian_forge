from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_transfer_request
def test_transfer_create_pre_v355(self):
    self.volume_client.api_version = api_versions.APIVersion('3.54')
    arglist = ['--no-snapshots', self.volume.id]
    verifylist = [('name', None), ('snapshots', False), ('volume', self.volume.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-volume-api-version 3.55 or greater is required', str(exc))