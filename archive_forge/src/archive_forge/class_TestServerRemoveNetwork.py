import collections
import copy
import getpass
import json
import tempfile
from unittest import mock
from unittest.mock import call
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
class TestServerRemoveNetwork(TestServer):

    def setUp(self):
        super(TestServerRemoveNetwork, self).setUp()
        self.cmd = server.RemoveNetwork(self.app, None)
        self.fake_inf = mock.Mock()
        self.methods = {'server_interfaces': [self.fake_inf], 'delete_server_interface': None}
        self.find_network = mock.Mock()
        self.app.client_manager.network.find_network = self.find_network
        self.compute_sdk_client.server_interfaces.return_value = [self.fake_inf]

    def _test_server_remove_network(self, network_id):
        self.fake_inf.net_id = network_id
        self.fake_inf.port_id = 'fake-port'
        servers = self.setup_sdk_servers_mock(count=1)
        network = 'fake-network'
        arglist = [servers[0].id, network]
        verifylist = [('server', servers[0].id), ('network', network)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.server_interfaces.assert_called_once_with(servers[0])
        self.compute_sdk_client.delete_server_interface.assert_called_once_with('fake-port', server=servers[0])
        self.assertIsNone(result)

    def test_server_remove_network(self):
        self._test_server_remove_network(self.find_network.return_value.id)
        self.find_network.assert_called_once_with('fake-network', ignore_missing=False)

    def test_server_remove_network_no_neutron(self):
        self.app.client_manager.network_endpoint_enabled = False
        self._test_server_remove_network('fake-network')
        self.find_network.assert_not_called()