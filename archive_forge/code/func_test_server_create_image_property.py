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
def test_server_create_image_property(self):
    arglist = ['--image-property', 'hypervisor_type=qemu', '--flavor', 'flavor1', self.new_server.name]
    verifylist = [('image_properties', {'hypervisor_type': 'qemu'}), ('flavor', 'flavor1'), ('config_drive', False), ('server_name', self.new_server.name)]
    image_info = {'hypervisor_type': 'qemu'}
    _image = image_fakes.create_one_image(image_info)
    self.image_client.images.return_value = [_image]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = dict(files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=None, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics=[], meta=None, scheduler_hints={}, config_drive=None)
    self.servers_mock.create.assert_called_with(self.new_server.name, _image, self.flavor, **kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist(), data)