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
def test_set_port_append_allowed_address_pair(self):
    _testport = network_fakes.create_one_port({'allowed_address_pairs': [{'ip_address': '192.168.1.123'}]})
    self.network_client.find_port = mock.Mock(return_value=_testport)
    arglist = ['--allowed-address', 'ip-address=192.168.1.45', _testport.name]
    verifylist = [('allowed_address_pairs', [{'ip-address': '192.168.1.45'}]), ('port', _testport.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'allowed_address_pairs': [{'ip_address': '192.168.1.123'}, {'ip_address': '192.168.1.45'}]}
    self.network_client.update_port.assert_called_once_with(_testport, **attrs)
    self.assertIsNone(result)