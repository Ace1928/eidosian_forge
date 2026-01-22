from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import security_group
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch('openstackclient.api.compute_v2.APIv2.security_group_list')
class TestListSecurityGroupCompute(compute_fakes.TestComputev2):
    _security_groups = compute_fakes.create_security_groups(count=3)
    columns = ('ID', 'Name', 'Description')
    columns_all_projects = ('ID', 'Name', 'Description', 'Project')
    data = []
    for grp in _security_groups:
        data.append((grp['id'], grp['name'], grp['description']))
    data_all_projects = []
    for grp in _security_groups:
        data_all_projects.append((grp['id'], grp['name'], grp['description'], grp['tenant_id']))

    def setUp(self):
        super(TestListSecurityGroupCompute, self).setUp()
        self.app.client_manager.network_endpoint_enabled = False
        self.cmd = security_group.ListSecurityGroup(self.app, None)

    def test_security_group_list_no_options(self, sg_mock):
        sg_mock.return_value = self._security_groups
        arglist = []
        verifylist = [('all_projects', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'search_opts': {'all_tenants': False}}
        sg_mock.assert_called_once_with(**kwargs)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_security_group_list_all_projects(self, sg_mock):
        sg_mock.return_value = self._security_groups
        arglist = ['--all-projects']
        verifylist = [('all_projects', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'search_opts': {'all_tenants': True}}
        sg_mock.assert_called_once_with(**kwargs)
        self.assertEqual(self.columns_all_projects, columns)
        self.assertCountEqual(self.data_all_projects, list(data))