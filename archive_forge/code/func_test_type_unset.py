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
def test_type_unset(self):
    arglist = ['--property', 'property', '--property', 'multi_property', self.volume_type.id]
    verifylist = [('properties', ['property', 'multi_property']), ('volume_type', self.volume_type.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.volume_type.unset_keys.assert_called_once_with(['property', 'multi_property'])
    self.assertIsNone(result)