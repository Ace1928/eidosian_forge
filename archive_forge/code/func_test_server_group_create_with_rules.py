from unittest import mock
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import server_group
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
def test_server_group_create_with_rules(self, sm_mock):
    arglist = ['--policy', 'soft-anti-affinity', '--rule', 'max_server_per_host=2', 'affinity_group']
    verifylist = [('policy', 'soft-anti-affinity'), ('rules', {'max_server_per_host': '2'}), ('name', 'affinity_group')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.create_server_group.assert_called_once_with(name=parsed_args.name, policy=parsed_args.policy, rules=parsed_args.rules)
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)