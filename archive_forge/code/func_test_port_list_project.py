from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.network.v2 import port
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_port_list_project(self):
    project = identity_fakes.FakeProject.create_one_project()
    self.projects_mock.get.return_value = project
    arglist = ['--project', project.id]
    verifylist = [('project', project.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    filters = {'project_id': project.id, 'fields': LIST_FIELDS_TO_RETRIEVE}
    self.network_client.ports.assert_called_once_with(**filters)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))