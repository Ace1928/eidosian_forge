from unittest import mock
from openstackclient.network.v2 import network_qos_rule_type as _qos_rule_type
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListNetworkQosRuleType(TestNetworkQosRuleType):
    qos_rule_types = network_fakes.FakeNetworkQosRuleType.create_qos_rule_types(count=3)
    columns = ('Type',)
    data = []
    for qos_rule_type in qos_rule_types:
        data.append((qos_rule_type.type,))

    def setUp(self):
        super(TestListNetworkQosRuleType, self).setUp()
        self.network_client.qos_rule_types = mock.Mock(return_value=self.qos_rule_types)
        self.cmd = _qos_rule_type.ListNetworkQosRuleType(self.app, self.namespace)

    def test_qos_rule_type_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.qos_rule_types.assert_called_once_with(**{})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_qos_rule_type_list_all_supported(self):
        arglist = ['--all-supported']
        verifylist = [('all_supported', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.qos_rule_types.assert_called_once_with(**{'all_supported': True})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_qos_rule_type_list_all_rules(self):
        arglist = ['--all-rules']
        verifylist = [('all_rules', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.qos_rule_types.assert_called_once_with(**{'all_rules': True})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))