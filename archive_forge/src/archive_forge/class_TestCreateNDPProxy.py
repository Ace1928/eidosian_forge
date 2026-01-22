from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import ndp_proxy
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestCreateNDPProxy(TestNDPProxy):

    def setUp(self):
        super(TestCreateNDPProxy, self).setUp()
        attrs = {'router_id': self.router.id, 'port_id': self.port.id}
        self.ndp_proxy = network_fakes.create_one_ndp_proxy(attrs)
        self.columns = ('created_at', 'description', 'id', 'ip_address', 'name', 'port_id', 'project_id', 'revision_number', 'router_id', 'updated_at')
        self.data = (self.ndp_proxy.created_at, self.ndp_proxy.description, self.ndp_proxy.id, self.ndp_proxy.ip_address, self.ndp_proxy.name, self.ndp_proxy.port_id, self.ndp_proxy.project_id, self.ndp_proxy.revision_number, self.ndp_proxy.router_id, self.ndp_proxy.updated_at)
        self.network_client.create_ndp_proxy = mock.Mock(return_value=self.ndp_proxy)
        self.cmd = ndp_proxy.CreateNDPProxy(self.app, self.namespace)

    def test_create_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_all_options(self):
        arglist = [self.ndp_proxy.router_id, '--name', self.ndp_proxy.name, '--port', self.ndp_proxy.port_id, '--ip-address', self.ndp_proxy.ip_address, '--description', self.ndp_proxy.description]
        verifylist = [('name', self.ndp_proxy.name), ('router', self.ndp_proxy.router_id), ('port', self.ndp_proxy.port_id), ('ip_address', self.ndp_proxy.ip_address), ('description', self.ndp_proxy.description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_ndp_proxy.assert_called_once_with(**{'name': self.ndp_proxy.name, 'router_id': self.ndp_proxy.router_id, 'ip_address': self.ndp_proxy.ip_address, 'port_id': self.ndp_proxy.port_id, 'description': self.ndp_proxy.description})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)