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
@mock.patch('openstackclient.compute.v2.server.os.system')
class TestServerSsh(TestServer):

    def setUp(self):
        super().setUp()
        self.cmd = server.SshServer(self.app, None)
        self.app.client_manager.auth_ref = mock.Mock()
        self.app.client_manager.auth_ref.username = 'cloud'
        self.attrs = {'addresses': {'public': [{'addr': '192.168.1.30', 'OS-EXT-IPS-MAC:mac_addr': '00:0c:29:0d:11:74', 'OS-EXT-IPS:type': 'fixed', 'version': 4}]}}
        self.server = compute_fakes.create_one_server(attrs=self.attrs, methods=self.methods)
        self.servers_mock.get.return_value = self.server

    def test_server_ssh_no_opts(self, mock_exec):
        arglist = [self.server.name]
        verifylist = [('server', self.server.name), ('login', None), ('port', None), ('identity', None), ('option', None), ('ipv4', False), ('ipv6', False), ('address_type', 'public'), ('verbose', False), ('ssh_args', [])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(self.cmd.log, 'warning') as mock_warning:
            result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        mock_exec.assert_called_once_with('ssh 192.168.1.30 -l cloud')
        mock_warning.assert_not_called()

    def test_server_ssh_passthrough_opts(self, mock_exec):
        arglist = [self.server.name, '--', '-l', 'username', '-p', '2222']
        verifylist = [('server', self.server.name), ('login', None), ('port', None), ('identity', None), ('option', None), ('ipv4', False), ('ipv6', False), ('address_type', 'public'), ('verbose', False), ('ssh_args', ['-l', 'username', '-p', '2222'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(self.cmd.log, 'warning') as mock_warning:
            result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        mock_exec.assert_called_once_with('ssh 192.168.1.30 -l username -p 2222')
        mock_warning.assert_not_called()

    def test_server_ssh_deprecated_opts(self, mock_exec):
        arglist = [self.server.name, '-l', 'username', '-p', '2222']
        verifylist = [('server', self.server.name), ('login', 'username'), ('port', 2222), ('identity', None), ('option', None), ('ipv4', False), ('ipv6', False), ('address_type', 'public'), ('verbose', False), ('ssh_args', [])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(self.cmd.log, 'warning') as mock_warning:
            result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        mock_exec.assert_called_once_with('ssh 192.168.1.30 -p 2222 -l username')
        mock_warning.assert_called_once()
        self.assertIn('The ssh options have been deprecated.', mock_warning.call_args[0][0])