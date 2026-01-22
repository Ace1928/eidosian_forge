from unittest import mock
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_access_rules as osc_share_access_rules
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.ddt
class TestShareAccessList(TestShareAccess):
    access_rules_columns = ['ID', 'Access Type', 'Access To', 'Access Level', 'State', 'Access Key', 'Created At', 'Updated At']

    def setUp(self):
        super(TestShareAccessList, self).setUp()
        self.share = manila_fakes.FakeShare.create_one_share()
        self.access_rule_1 = manila_fakes.FakeShareAccessRule.create_one_access_rule(attrs={'share_id': self.share.id})
        self.access_rule_2 = manila_fakes.FakeShareAccessRule.create_one_access_rule(attrs={'share_id': self.share.id, 'access_to': 'admin'})
        self.access_rules = [self.access_rule_1, self.access_rule_2]
        self.shares_mock.get.return_value = self.share
        self.access_rules_mock.access_list.return_value = self.access_rules
        self.values_list = (oscutils.get_dict_properties(a._info, self.access_rules_columns) for a in self.access_rules)
        self.cmd = osc_share_access_rules.ListShareAccess(self.app, None)

    def test_access_rules_list(self):
        arglist = [self.share.id]
        verifylist = [('share', self.share.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.get.assert_called_with(self.share.id)
        self.access_rules_mock.access_list.assert_called_with(self.share, {})
        self.assertEqual(self.access_rules_columns, columns)
        self.assertEqual(tuple(self.values_list), tuple(data))

    def test_access_rules_list_filter_properties(self):
        arglist = [self.share.id, '--properties', 'key=value']
        verifylist = [('share', self.share.id), ('properties', ['key=value'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.get.assert_called_with(self.share.id)
        self.access_rules_mock.access_list.assert_called_with(self.share, {'metadata': {'key': 'value'}})
        self.assertEqual(self.access_rules_columns, columns)
        self.assertEqual(tuple(self.values_list), tuple(data))

    @ddt.data({'access_to': '10.0.0.0/0', 'access_type': 'ip'}, {'access_key': '10.0.0.0/0', 'access_level': 'rw'})
    def test_access_rules_list_access_filters(self, filters):
        arglist = [self.share.id]
        verifylist = [('share', self.share.id)]
        for filter_key, filter_value in filters.items():
            filter_arg = filter_key.replace('_', '-')
            arglist.append(f'--{filter_arg}')
            arglist.append(filter_value)
            verifylist.append((filter_key, filter_value))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.get.assert_called_with(self.share.id)
        self.access_rules_mock.access_list.assert_called_with(self.share, filters)
        self.assertEqual(self.access_rules_columns, columns)
        self.assertEqual(tuple(self.values_list), tuple(data))

    @ddt.data({'access_to': '10.0.0.0/0', 'access_type': 'ip'}, {'access_key': '10.0.0.0/0', 'access_level': 'rw'})
    def test_access_rules_list_access_filters_command_error(self, filters):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.81')
        arglist = [self.share.id]
        verifylist = [('share', self.share.id)]
        for filter_key, filter_value in filters.items():
            filter_arg = filter_key.replace('_', '-')
            arglist.append(f'--{filter_arg}')
            arglist.append(filter_value)
            verifylist.append((filter_key, filter_value))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)