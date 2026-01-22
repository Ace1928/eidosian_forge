from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import security_group
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch('openstackclient.api.compute_v2.APIv2.security_group_find')
class TestShowSecurityGroupCompute(compute_fakes.TestComputev2):
    _security_group_rule = compute_fakes.create_one_security_group_rule()
    _security_group = compute_fakes.create_one_security_group(attrs={'rules': [_security_group_rule]})
    columns = ('description', 'id', 'name', 'project_id', 'rules')
    data = (_security_group['description'], _security_group['id'], _security_group['name'], _security_group['tenant_id'], security_group.ComputeSecurityGroupRulesColumn([_security_group_rule]))

    def setUp(self):
        super(TestShowSecurityGroupCompute, self).setUp()
        self.app.client_manager.network_endpoint_enabled = False
        self.cmd = security_group.ShowSecurityGroup(self.app, None)

    def test_security_group_show_no_options(self, sg_mock):
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, [], [])

    def test_security_group_show_all_options(self, sg_mock):
        sg_mock.return_value = self._security_group
        arglist = [self._security_group['id']]
        verifylist = [('group', self._security_group['id'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        sg_mock.assert_called_once_with(self._security_group['id'])
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)