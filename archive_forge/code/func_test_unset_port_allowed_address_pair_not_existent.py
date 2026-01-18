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
def test_unset_port_allowed_address_pair_not_existent(self):
    _fake_port = network_fakes.create_one_port({'allowed_address_pairs': [{'ip_address': '192.168.1.123'}]})
    self.network_client.find_port = mock.Mock(return_value=_fake_port)
    arglist = ['--allowed-address', 'ip-address=192.168.1.45', _fake_port.name]
    verifylist = [('allowed_address_pairs', [{'ip-address': '192.168.1.45'}])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)