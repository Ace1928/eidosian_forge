from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import local_ip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListLocalIP(TestLocalIP):
    local_ips = network_fakes.create_local_ips(count=3)
    fake_network = network_fakes.create_one_network({'id': 'fake_network_id'})
    columns = ('ID', 'Name', 'Description', 'Project', 'Local Port ID', 'Network', 'Local IP address', 'IP mode')
    data = []
    for lip in local_ips:
        data.append((lip.id, lip.name, lip.description, lip.project_id, lip.local_port_id, lip.network_id, lip.local_ip_address, lip.ip_mode))

    def setUp(self):
        super().setUp()
        self.network_client.local_ips = mock.Mock(return_value=self.local_ips)
        self.network_client.find_network = mock.Mock(return_value=self.fake_network)
        self.cmd = local_ip.ListLocalIP(self.app, self.namespace)

    def test_local_ip_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.local_ips.assert_called_once_with(**{})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_local_ip_list_name(self):
        arglist = ['--name', self.local_ips[0].name]
        verifylist = [('name', self.local_ips[0].name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.local_ips.assert_called_once_with(**{'name': self.local_ips[0].name})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_local_ip_list_project(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id]
        verifylist = [('project', project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.local_ips.assert_called_once_with(**{'project_id': project.id})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_local_ip_project_domain(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id, '--project-domain', project.domain_id]
        verifylist = [('project', project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'project_id': project.id}
        self.network_client.local_ips.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_local_ip_list_network(self):
        arglist = ['--network', 'fake_network_id']
        verifylist = [('network', 'fake_network_id')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.local_ips.assert_called_once_with(**{'network_id': 'fake_network_id'})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_local_ip_list_local_ip_address(self):
        arglist = ['--local-ip-address', self.local_ips[0].local_ip_address]
        verifylist = [('local_ip_address', self.local_ips[0].local_ip_address)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.local_ips.assert_called_once_with(**{'local_ip_address': self.local_ips[0].local_ip_address})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_local_ip_list_ip_mode(self):
        arglist = ['--ip-mode', self.local_ips[0].ip_mode]
        verifylist = [('ip_mode', self.local_ips[0].ip_mode)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.local_ips.assert_called_once_with(**{'ip_mode': self.local_ips[0].ip_mode})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))