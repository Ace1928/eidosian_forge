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
def test_unset_security_group(self):
    _fake_sg1 = network_fakes.FakeSecurityGroup.create_one_security_group()
    _fake_sg2 = network_fakes.FakeSecurityGroup.create_one_security_group()
    _fake_port = network_fakes.create_one_port({'security_group_ids': [_fake_sg1.id, _fake_sg2.id]})
    self.network_client.find_port = mock.Mock(return_value=_fake_port)
    self.network_client.find_security_group = mock.Mock(return_value=_fake_sg2)
    arglist = ['--security-group', _fake_sg2.id, _fake_port.name]
    verifylist = [('security_group_ids', [_fake_sg2.id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'security_group_ids': [_fake_sg1.id]}
    self.network_client.update_port.assert_called_once_with(_fake_port, **attrs)
    self.assertIsNone(result)