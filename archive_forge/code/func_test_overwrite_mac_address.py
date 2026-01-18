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
def test_overwrite_mac_address(self):
    _testport = network_fakes.create_one_port({'mac_address': '11:22:33:44:55:66'})
    self.network_client.find_port = mock.Mock(return_value=_testport)
    arglist = ['--mac-address', '66:55:44:33:22:11', _testport.name]
    verifylist = [('mac_address', '66:55:44:33:22:11')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'mac_address': '66:55:44:33:22:11'}
    self.network_client.update_port.assert_called_once_with(_testport, **attrs)
    self.assertIsNone(result)