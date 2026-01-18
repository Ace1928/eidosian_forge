from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import security_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_security_group_list_project(self):
    project = identity_fakes.FakeProject.create_one_project()
    self.projects_mock.get.return_value = project
    arglist = ['--project', project.id]
    verifylist = [('project', project.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    filters = {'project_id': project.id, 'fields': security_group.ListSecurityGroup.FIELDS_TO_RETRIEVE}
    self.network_client.security_groups.assert_called_once_with(**filters)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))