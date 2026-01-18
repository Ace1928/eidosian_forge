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
def test_server_create_with_network_tag(self):
    self.compute_client.api_version = api_versions.APIVersion('2.43')
    arglist = ['--image', 'image1', '--flavor', 'flavor1', '--nic', 'net-id=net1,tag=foo', self.new_server.name]
    verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('nics', [{'net-id': 'net1', 'port-id': '', 'v4-fixed-ip': '', 'v6-fixed-ip': '', 'tag': 'foo'}]), ('server_name', self.new_server.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    _network = mock.Mock(id='net1_uuid')
    self.network_client.find_network.return_value = _network
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics=[{'net-id': 'net1_uuid', 'v4-fixed-ip': '', 'v6-fixed-ip': '', 'port-id': '', 'tag': 'foo'}], scheduler_hints={}, config_drive=None)
    self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist(), data)
    self.network_client.find_network.assert_called_once()
    self.app.client_manager.network.find_network.assert_called_once()