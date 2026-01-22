from unittest import mock
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import server_group
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestServerGroupCreate(TestServerGroup):

    def setUp(self):
        super().setUp()
        self.compute_sdk_client.create_server_group.return_value = self.fake_server_group
        self.cmd = server_group.CreateServerGroup(self.app, None)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_server_group_create(self, sm_mock):
        arglist = ['--policy', 'anti-affinity', 'affinity_group']
        verifylist = [('policy', 'anti-affinity'), ('name', 'affinity_group')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.create_server_group.assert_called_once_with(name=parsed_args.name, policy=parsed_args.policy)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_server_group_create_with_soft_policies(self, sm_mock):
        arglist = ['--policy', 'soft-anti-affinity', 'affinity_group']
        verifylist = [('policy', 'soft-anti-affinity'), ('name', 'affinity_group')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.create_server_group.assert_called_once_with(name=parsed_args.name, policy=parsed_args.policy)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
    def test_server_group_create_with_soft_policies_pre_v215(self, sm_mock):
        arglist = ['--policy', 'soft-anti-affinity', 'affinity_group']
        verifylist = [('policy', 'soft-anti-affinity'), ('name', 'affinity_group')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.15 or greater is required', str(ex))

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_server_group_create_with_rules(self, sm_mock):
        arglist = ['--policy', 'soft-anti-affinity', '--rule', 'max_server_per_host=2', 'affinity_group']
        verifylist = [('policy', 'soft-anti-affinity'), ('rules', {'max_server_per_host': '2'}), ('name', 'affinity_group')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.create_server_group.assert_called_once_with(name=parsed_args.name, policy=parsed_args.policy, rules=parsed_args.rules)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    @mock.patch.object(sdk_utils, 'supports_microversion', side_effect=[True, False])
    def test_server_group_create_with_rules_pre_v264(self, sm_mock):
        arglist = ['--policy', 'soft-anti-affinity', '--rule', 'max_server_per_host=2', 'affinity_group']
        verifylist = [('policy', 'soft-anti-affinity'), ('rules', {'max_server_per_host': '2'}), ('name', 'affinity_group')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.64 or greater is required', str(ex))