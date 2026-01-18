import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import qos_specs
def test_qos_delete_with_force(self):
    arglist = ['--force', self.qos_specs[0].id]
    verifylist = [('force', True), ('qos_specs', [self.qos_specs[0].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.qos_mock.delete.assert_called_with(self.qos_specs[0].id, True)
    self.assertIsNone(result)