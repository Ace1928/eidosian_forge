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
def test_set_port_security_group_replace(self):
    sg1 = network_fakes.FakeSecurityGroup.create_one_security_group()
    sg2 = network_fakes.FakeSecurityGroup.create_one_security_group()
    _testport = network_fakes.create_one_port({'security_group_ids': [sg1.id]})
    self.network_client.find_port = mock.Mock(return_value=_testport)
    self.network_client.find_security_group = mock.Mock(return_value=sg2)
    arglist = ['--security-group', sg2.id, '--no-security-group', _testport.name]
    verifylist = [('security_group', [sg2.id]), ('no_security_group', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'security_group_ids': [sg2.id]}
    self.network_client.update_port.assert_called_once_with(_testport, **attrs)
    self.assertIsNone(result)