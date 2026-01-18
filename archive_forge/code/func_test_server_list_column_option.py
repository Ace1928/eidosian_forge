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
def test_server_list_column_option(self):
    arglist = ['-c', 'Project ID', '-c', 'User ID', '-c', 'Created At', '-c', 'Security Groups', '-c', 'Task State', '-c', 'Power State', '-c', 'Image ID', '-c', 'Flavor ID', '-c', 'Availability Zone', '-c', 'Host', '-c', 'Properties', '--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
    self.assertIn('Project ID', columns)
    self.assertIn('User ID', columns)
    self.assertIn('Created At', columns)
    self.assertIn('Security Groups', columns)
    self.assertIn('Task State', columns)
    self.assertIn('Power State', columns)
    self.assertIn('Image ID', columns)
    self.assertIn('Flavor ID', columns)
    self.assertIn('Availability Zone', columns)
    self.assertIn('Host', columns)
    self.assertIn('Properties', columns)
    self.assertCountEqual(columns, set(columns))