from unittest import mock
from openstackclient.identity.v3 import endpoint_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestEndpointGroupList(TestEndpointGroup):
    endpoint_group = identity_fakes.FakeEndpointGroup.create_one_endpointgroup()
    project = identity_fakes.FakeProject.create_one_project()
    domain = identity_fakes.FakeDomain.create_one_domain()
    columns = ('ID', 'Name', 'Description')

    def setUp(self):
        super(TestEndpointGroupList, self).setUp()
        self.endpoint_groups_mock.list.return_value = [self.endpoint_group]
        self.endpoint_groups_mock.get.return_value = self.endpoint_group
        self.epf_mock.list_projects_for_endpoint_group.return_value = [self.project]
        self.epf_mock.list_endpoint_groups_for_project.return_value = [self.endpoint_group]
        self.cmd = endpoint_group.ListEndpointGroup(self.app, None)

    def test_endpoint_group_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.endpoint_groups_mock.list.assert_called_with()
        self.assertEqual(self.columns, columns)
        datalist = ((self.endpoint_group.id, self.endpoint_group.name, self.endpoint_group.description),)
        self.assertEqual(datalist, tuple(data))

    def test_endpoint_group_list_projects_by_endpoint_group(self):
        arglist = ['--endpointgroup', self.endpoint_group.id]
        verifylist = [('endpointgroup', self.endpoint_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.epf_mock.list_projects_for_endpoint_group.assert_called_with(endpoint_group=self.endpoint_group.id)
        self.assertEqual(self.columns, columns)
        datalist = ((self.project.id, self.project.name, self.project.description),)
        self.assertEqual(datalist, tuple(data))

    def test_endpoint_group_list_by_project(self):
        self.epf_mock.list_endpoints_for_project.return_value = [self.endpoint_group]
        self.projects_mock.get.return_value = self.project
        arglist = ['--project', self.project.name, '--domain', self.domain.name]
        verifylist = [('project', self.project.name), ('domain', self.domain.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.epf_mock.list_endpoint_groups_for_project.assert_called_with(project=self.project.id)
        self.assertEqual(self.columns, columns)
        datalist = ((self.endpoint_group.id, self.endpoint_group.name, self.endpoint_group.description),)
        self.assertEqual(datalist, tuple(data))