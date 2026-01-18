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
def test_server_create_with_conflict_network_options(self):
    arglist = ['--image', 'image1', '--flavor', 'flavor1', '--nic', 'none', '--nic', 'auto', '--nic', 'port-id=port1', self.new_server.name]
    verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('nics', ['none', 'auto', {'net-id': '', 'port-id': 'port1', 'v4-fixed-ip': '', 'v6-fixed-ip': ''}]), ('config_drive', False), ('server_name', self.new_server.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    get_endpoints = mock.Mock()
    get_endpoints.return_value = {'network': []}
    self.app.client_manager.auth_ref = mock.Mock()
    self.app.client_manager.auth_ref.service_catalog = mock.Mock()
    self.app.client_manager.auth_ref.service_catalog.get_endpoints = get_endpoints
    port_resource = mock.Mock(id='port1_uuid')
    self.network_client.find_port.return_value = port_resource
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertNotCalled(self.servers_mock.create)