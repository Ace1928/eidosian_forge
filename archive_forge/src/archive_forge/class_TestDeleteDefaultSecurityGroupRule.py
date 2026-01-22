from unittest import mock
from unittest.mock import call
import uuid
from openstack.network.v2 import _proxy
from openstack.network.v2 import (
from openstack.test import fakes as sdk_fakes
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import default_security_group_rule
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestDeleteDefaultSecurityGroupRule(TestDefaultSecurityGroupRule):
    default_security_group_rule_attrs = {'direction': 'ingress', 'ether_type': 'IPv4', 'port_range_max': None, 'port_range_min': None, 'protocol': None, 'remote_group_id': None, 'remote_address_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'location': 'MUNCHMUNCHMUNCH', 'used_in_default_sg': False, 'used_in_non_default_sg': True}
    _default_sg_rules = list(sdk_fakes.generate_fake_resources(_default_security_group_rule.DefaultSecurityGroupRule, count=2, attrs=default_security_group_rule_attrs))

    def setUp(self):
        super(TestDeleteDefaultSecurityGroupRule, self).setUp()
        self.sdk_client.delete_default_security_group_rule.return_value = None
        self.cmd = default_security_group_rule.DeleteDefaultSecurityGroupRule(self.app, self.namespace)

    def test_default_security_group_rule_delete(self):
        arglist = [self._default_sg_rules[0].id]
        verifylist = [('rule', [self._default_sg_rules[0].id])]
        self.sdk_client.find_default_security_group_rule.return_value = self._default_sg_rules[0]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.sdk_client.delete_default_security_group_rule.assert_called_once_with(self._default_sg_rules[0])
        self.assertIsNone(result)

    def test_multi_default_security_group_rules_delete(self):
        arglist = []
        verifylist = []
        for s in self._default_sg_rules:
            arglist.append(s.id)
        verifylist = [('rule', arglist)]
        self.sdk_client.find_default_security_group_rule.side_effect = self._default_sg_rules
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for s in self._default_sg_rules:
            calls.append(call(s))
        self.sdk_client.delete_default_security_group_rule.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_multi_default_security_group_rules_delete_with_exception(self):
        arglist = [self._default_sg_rules[0].id, 'unexist_rule']
        verifylist = [('rule', [self._default_sg_rules[0].id, 'unexist_rule'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        find_mock_result = [self._default_sg_rules[0], exceptions.CommandError]
        self.sdk_client.find_default_security_group_rule = mock.Mock(side_effect=find_mock_result)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 default rules failed to delete.', str(e))
        self.sdk_client.find_default_security_group_rule.assert_any_call(self._default_sg_rules[0].id, ignore_missing=False)
        self.sdk_client.find_default_security_group_rule.assert_any_call('unexist_rule', ignore_missing=False)
        self.sdk_client.delete_default_security_group_rule.assert_called_once_with(self._default_sg_rules[0])