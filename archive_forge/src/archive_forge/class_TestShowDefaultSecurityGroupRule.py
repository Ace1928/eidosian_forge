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
class TestShowDefaultSecurityGroupRule(TestDefaultSecurityGroupRule):
    _default_sg_rule = sdk_fakes.generate_fake_resource(_default_security_group_rule.DefaultSecurityGroupRule)
    columns = ('description', 'direction', 'ether_type', 'id', 'port_range_max', 'port_range_min', 'protocol', 'remote_address_group_id', 'remote_group_id', 'remote_ip_prefix', 'used_in_default_sg', 'used_in_non_default_sg')
    data = (_default_sg_rule.description, _default_sg_rule.direction, _default_sg_rule.ether_type, _default_sg_rule.id, _default_sg_rule.port_range_max, _default_sg_rule.port_range_min, _default_sg_rule.protocol, _default_sg_rule.remote_address_group_id, _default_sg_rule.remote_group_id, _default_sg_rule.remote_ip_prefix, _default_sg_rule.used_in_default_sg, _default_sg_rule.used_in_non_default_sg)

    def setUp(self):
        super(TestShowDefaultSecurityGroupRule, self).setUp()
        self.sdk_client.find_default_security_group_rule.return_value = self._default_sg_rule
        self.cmd = default_security_group_rule.ShowDefaultSecurityGroupRule(self.app, self.namespace)

    def test_show_no_options(self):
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, [], [])

    def test_show_all_options(self):
        arglist = [self._default_sg_rule.id]
        verifylist = [('rule', self._default_sg_rule.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sdk_client.find_default_security_group_rule.assert_called_once_with(self._default_sg_rule.id, ignore_missing=False)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)