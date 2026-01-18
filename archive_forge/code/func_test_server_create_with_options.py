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
def test_server_create_with_options(self):
    arglist = ['--image', 'image1', '--flavor', 'flavor1', '--key-name', 'keyname', '--property', 'Beta=b', '--security-group', 'securitygroup', '--use-config-drive', '--password', 'passw0rd', '--hint', 'a=b', '--hint', 'a=c', '--server-group', 'servergroup', self.new_server.name]
    verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('key_name', 'keyname'), ('properties', {'Beta': 'b'}), ('security_group', ['securitygroup']), ('hints', {'a': ['b', 'c']}), ('server_group', 'servergroup'), ('config_drive', True), ('password', 'passw0rd'), ('server_name', self.new_server.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    fake_server_group = compute_fakes.create_one_server_group()
    self.compute_client.server_groups.get.return_value = fake_server_group
    fake_sg = network_fakes.FakeSecurityGroup.create_security_groups()
    mock_find_sg = network_fakes.FakeSecurityGroup.get_security_groups(fake_sg)
    self.app.client_manager.network.find_security_group = mock_find_sg
    columns, data = self.cmd.take_action(parsed_args)
    mock_find_sg.assert_called_once_with('securitygroup', ignore_missing=False)
    kwargs = dict(meta={'Beta': 'b'}, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[fake_sg[0].id], userdata=None, key_name='keyname', availability_zone=None, admin_pass='passw0rd', block_device_mapping_v2=[], nics=[], scheduler_hints={'a': ['b', 'c'], 'group': fake_server_group.id}, config_drive=True)
    self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist(), data)