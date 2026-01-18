import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v1 import fakes as volume_fakes
from openstackclient.volume.v1 import qos_specs
def test_qos_delete_with_name(self):
    arglist = [self.qos_specs[0].name]
    verifylist = [('qos_specs', [self.qos_specs[0].name])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.qos_mock.delete.assert_called_with(self.qos_specs[0].id, False)
    self.assertIsNone(result)