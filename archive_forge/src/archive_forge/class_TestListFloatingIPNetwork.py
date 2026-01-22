from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip as fip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListFloatingIPNetwork(TestFloatingIPNetwork):
    floating_ips = network_fakes.FakeFloatingIP.create_floating_ips(count=3)
    fake_network = network_fakes.create_one_network({'id': 'fake_network_id'})
    fake_port = network_fakes.create_one_port({'id': 'fake_port_id'})
    fake_router = network_fakes.FakeRouter.create_one_router({'id': 'fake_router_id'})
    columns = ('ID', 'Floating IP Address', 'Fixed IP Address', 'Port', 'Floating Network', 'Project')
    columns_long = columns + ('Router', 'Status', 'Description', 'Tags', 'DNS Name', 'DNS Domain')
    data = []
    data_long = []
    for ip in floating_ips:
        data.append((ip.id, ip.floating_ip_address, ip.fixed_ip_address, ip.port_id, ip.floating_network_id, ip.project_id))
        data_long.append((ip.id, ip.floating_ip_address, ip.fixed_ip_address, ip.port_id, ip.floating_network_id, ip.project_id, ip.router_id, ip.status, ip.description, ip.tags, ip.dns_domain, ip.dns_name))

    def setUp(self):
        super(TestListFloatingIPNetwork, self).setUp()
        self.network_client.ips = mock.Mock(return_value=self.floating_ips)
        self.network_client.find_network = mock.Mock(return_value=self.fake_network)
        self.network_client.find_port = mock.Mock(return_value=self.fake_port)
        self.network_client.find_router = mock.Mock(return_value=self.fake_router)
        self.cmd = fip.ListFloatingIP(self.app, self.namespace)

    def test_floating_ip_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ips.assert_called_once_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_floating_ip_list_network(self):
        arglist = ['--network', 'fake_network_id']
        verifylist = [('network', 'fake_network_id')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ips.assert_called_once_with(**{'floating_network_id': 'fake_network_id'})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_floating_ip_list_port(self):
        arglist = ['--port', 'fake_port_id']
        verifylist = [('port', 'fake_port_id')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ips.assert_called_once_with(**{'port_id': 'fake_port_id'})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_floating_ip_list_fixed_ip_address(self):
        arglist = ['--fixed-ip-address', self.floating_ips[0].fixed_ip_address]
        verifylist = [('fixed_ip_address', self.floating_ips[0].fixed_ip_address)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ips.assert_called_once_with(**{'fixed_ip_address': self.floating_ips[0].fixed_ip_address})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_floating_ip_list_floating_ip_address(self):
        arglist = ['--floating-ip-address', self.floating_ips[0].floating_ip_address]
        verifylist = [('floating_ip_address', self.floating_ips[0].floating_ip_address)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ips.assert_called_once_with(**{'floating_ip_address': self.floating_ips[0].floating_ip_address})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_floating_ip_list_long(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ips.assert_called_once_with()
        self.assertEqual(self.columns_long, columns)
        self.assertEqual(self.data_long, list(data))

    def test_floating_ip_list_status(self):
        arglist = ['--status', 'ACTIVE', '--long']
        verifylist = [('status', 'ACTIVE')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ips.assert_called_once_with(**{'status': 'ACTIVE'})
        self.assertEqual(self.columns_long, columns)
        self.assertEqual(self.data_long, list(data))

    def test_floating_ip_list_project(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id]
        verifylist = [('project', project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'project_id': project.id}
        self.network_client.ips.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_floating_ip_list_project_domain(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id, '--project-domain', project.domain_id]
        verifylist = [('project', project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'project_id': project.id}
        self.network_client.ips.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_floating_ip_list_router(self):
        arglist = ['--router', 'fake_router_id', '--long']
        verifylist = [('router', 'fake_router_id')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ips.assert_called_once_with(**{'router_id': 'fake_router_id'})
        self.assertEqual(self.columns_long, columns)
        self.assertEqual(self.data_long, list(data))

    def test_list_with_tag_options(self):
        arglist = ['--tags', 'red,blue', '--any-tags', 'red,green', '--not-tags', 'orange,yellow', '--not-any-tags', 'black,white']
        verifylist = [('tags', ['red', 'blue']), ('any_tags', ['red', 'green']), ('not_tags', ['orange', 'yellow']), ('not_any_tags', ['black', 'white'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ips.assert_called_once_with(**{'tags': 'red,blue', 'any_tags': 'red,green', 'not_tags': 'orange,yellow', 'not_any_tags': 'black,white'})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))