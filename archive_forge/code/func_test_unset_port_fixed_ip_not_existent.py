from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.network.v2 import port
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_unset_port_fixed_ip_not_existent(self):
    arglist = ['--fixed-ip', 'ip-address=1.0.0.1', '--binding-profile', 'Superman', self._testport.name]
    verifylist = [('fixed_ip', [{'ip-address': '1.0.0.1'}]), ('binding_profile', ['Superman'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)