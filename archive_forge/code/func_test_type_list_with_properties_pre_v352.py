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
def test_type_list_with_properties_pre_v352(self):
    self.app.client_manager.volume.api_version = api_versions.APIVersion('3.51')
    arglist = ['--property', 'foo=bar']
    verifylist = [('encryption_type', False), ('long', False), ('is_public', None), ('default', False), ('properties', {'foo': 'bar'})]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-volume-api-version 3.52 or greater is required', str(exc))