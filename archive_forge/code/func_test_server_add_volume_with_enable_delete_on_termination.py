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
@mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
def test_server_add_volume_with_enable_delete_on_termination(self, sm_mock):
    self.volume_attachment.delete_on_termination = True
    arglist = ['--enable-delete-on-termination', '--device', '/dev/sdb', self.servers[0].id, self.volumes[0].id]
    verifylist = [('server', self.servers[0].id), ('volume', self.volumes[0].id), ('device', '/dev/sdb'), ('enable_delete_on_termination', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    expected_columns = ('ID', 'Server ID', 'Volume ID', 'Device', 'Tag', 'Delete On Termination')
    expected_data = (self.volume_attachment.id, self.volume_attachment.server_id, self.volume_attachment.volume_id, self.volume_attachment.device, self.volume_attachment.tag, self.volume_attachment.delete_on_termination)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(expected_columns, columns)
    self.assertEqual(expected_data, data)
    self.compute_sdk_client.create_volume_attachment.assert_called_once_with(self.servers[0], volumeId=self.volumes[0].id, device='/dev/sdb', delete_on_termination=True)