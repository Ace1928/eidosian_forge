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
def test_server_add_floating_ip_with_fixed_ip_no_port_found(self):
    _server = compute_fakes.create_one_server()
    self.servers_mock.get.return_value = _server
    _port = network_fakes.create_one_port()
    _floating_ip = network_fakes.FakeFloatingIP.create_one_floating_ip()
    self.network_client.find_ip = mock.Mock(return_value=_floating_ip)
    self.network_client.ports = mock.Mock(return_value=[_port])
    nonexistent_ip = '10.0.0.9'
    arglist = ['--fixed-ip-address', nonexistent_ip, _server.id, _floating_ip['floating_ip_address']]
    verifylist = [('fixed_ip_address', nonexistent_ip), ('server', _server.id), ('ip_address', _floating_ip['floating_ip_address'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.network_client.find_ip.assert_called_once_with(_floating_ip['floating_ip_address'], ignore_missing=False)
    self.network_client.ports.assert_called_once_with(device_id=_server.id)
    self.network_client.update_ip.assert_not_called()