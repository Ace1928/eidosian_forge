from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import address_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListAddressGroup(TestAddressGroup):
    address_groups = network_fakes.create_address_groups(count=3)
    columns = ('ID', 'Name', 'Description', 'Project', 'Addresses')
    data = []
    for group in address_groups:
        data.append((group.id, group.name, group.description, group.project_id, group.addresses))

    def setUp(self):
        super(TestListAddressGroup, self).setUp()
        self.network_client.address_groups = mock.Mock(return_value=self.address_groups)
        self.cmd = address_group.ListAddressGroup(self.app, self.namespace)

    def test_address_group_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.address_groups.assert_called_once_with(**{})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_address_group_list_name(self):
        arglist = ['--name', self.address_groups[0].name]
        verifylist = [('name', self.address_groups[0].name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.address_groups.assert_called_once_with(**{'name': self.address_groups[0].name})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_address_group_list_project(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id]
        verifylist = [('project', project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.address_groups.assert_called_once_with(project_id=project.id)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_address_group_project_domain(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id, '--project-domain', project.domain_id]
        verifylist = [('project', project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.address_groups.assert_called_once_with(project_id=project.id)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))