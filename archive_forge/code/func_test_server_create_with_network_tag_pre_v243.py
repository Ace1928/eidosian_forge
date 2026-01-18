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
def test_server_create_with_network_tag_pre_v243(self):
    self.compute_client.api_version = api_versions.APIVersion('2.42')
    arglist = ['--image', 'image1', '--flavor', 'flavor1', '--nic', 'net-id=net1,tag=foo', self.new_server.name]
    verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('nics', [{'net-id': 'net1', 'port-id': '', 'v4-fixed-ip': '', 'v6-fixed-ip': '', 'tag': 'foo'}]), ('server_name', self.new_server.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)