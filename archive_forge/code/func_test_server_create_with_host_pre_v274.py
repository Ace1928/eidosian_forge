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
def test_server_create_with_host_pre_v274(self):
    self.compute_client.api_version = 2.73
    arglist = ['--image', 'image1', '--flavor', 'flavor1', '--host', 'host1', self.new_server.name]
    verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('host', 'host1'), ('config_drive', False), ('server_name', self.new_server.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch.object(api_versions, 'APIVersion', return_value=2.74):
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)