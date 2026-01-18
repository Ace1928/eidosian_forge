from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_policy
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_qos_policy_list_no_share(self):
    arglist = ['--no-share']
    verifylist = [('no_share', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.qos_policies.assert_called_once_with(**{'shared': False})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, list(data))