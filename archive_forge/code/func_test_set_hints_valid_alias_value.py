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
def test_set_hints_valid_alias_value(self):
    testport = network_fakes.create_one_port()
    self.network_client.find_port = mock.Mock(return_value=testport)
    self.network_client.find_extension = mock.Mock(return_value=['port-hints', 'port-hint-ovs-tx-steering'])
    arglist = ['--hint', 'ovs-tx-steering=hash', testport.name]
    verifylist = [('hint', {'ovs-tx-steering': 'hash'}), ('port', testport.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.update_port.assert_called_once_with(testport, **{'hints': {'openvswitch': {'other_config': {'tx-steering': 'hash'}}}})
    self.assertIsNone(result)