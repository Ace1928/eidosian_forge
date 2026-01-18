from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v1 import fakes as volume_fakes
from openstackclient.volume.v1 import volume_type
def test_type_unset_nothing(self):
    arglist = [self.volume_type.id]
    verifylist = [('volume_type', self.volume_type.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertIsNone(result)