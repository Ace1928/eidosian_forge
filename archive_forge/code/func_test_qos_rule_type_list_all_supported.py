from unittest import mock
from openstackclient.network.v2 import network_qos_rule_type as _qos_rule_type
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_qos_rule_type_list_all_supported(self):
    arglist = ['--all-supported']
    verifylist = [('all_supported', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.qos_rule_types.assert_called_once_with(**{'all_supported': True})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, list(data))