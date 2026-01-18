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
def test_server_list_long_with_host_status_v216(self):
    self._set_mock_microversion('2.16')
    self.data1 = tuple(((s.id, s.name, s.status, getattr(s, 'task_state'), server.PowerStateColumn(getattr(s, 'power_state')), server.AddressesColumn(s.addresses), self.image.name if s.image else server.IMAGE_STRING_FOR_BFV, s.image['id'] if s.image else server.IMAGE_STRING_FOR_BFV, self.flavor.name, s.flavor['id'], getattr(s, 'availability_zone'), server.HostColumn(getattr(s, 'hypervisor_hostname')), format_columns.DictColumn(s.metadata)) for s in self.servers))
    arglist = ['--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
    self.assertEqual(self.columns_long, columns)
    self.assertEqual(tuple(self.data1), tuple(data))
    self.compute_sdk_client.servers.reset_mock()
    self.attrs['host_status'] = 'UP'
    servers = self.setup_sdk_servers_mock(3)
    self.compute_sdk_client.servers.return_value = servers
    Image = collections.namedtuple('Image', 'id name')
    self.image_client.images.return_value = [Image(id=s.image['id'], name=self.image.name) for s in servers if s.image]
    columns_long = self.columns_long + ('Host Status',)
    self.data2 = tuple(((s.id, s.name, s.status, getattr(s, 'task_state'), server.PowerStateColumn(getattr(s, 'power_state')), server.AddressesColumn(s.addresses), self.image.name if s.image else server.IMAGE_STRING_FOR_BFV, s.image['id'] if s.image else server.IMAGE_STRING_FOR_BFV, self.flavor.name, s.flavor['id'], getattr(s, 'availability_zone'), server.HostColumn(getattr(s, 'hypervisor_hostname')), format_columns.DictColumn(s.metadata), s.host_status) for s in servers))
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
    self.assertEqual(columns_long, columns)
    self.assertEqual(tuple(self.data2), tuple(data))