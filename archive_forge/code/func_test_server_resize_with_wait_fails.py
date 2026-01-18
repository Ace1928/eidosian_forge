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
@mock.patch.object(common_utils, 'wait_for_status', return_value=False)
def test_server_resize_with_wait_fails(self, mock_wait_for_status):
    arglist = ['--flavor', self.flavors_get_return_value.id, '--wait', self.server.id]
    verifylist = [('flavor', self.flavors_get_return_value.id), ('confirm', False), ('revert', False), ('wait', True), ('server', self.server.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.servers_mock.get.assert_called_with(self.server.id)
    kwargs = dict(success_status=['active', 'verify_resize'])
    mock_wait_for_status.assert_called_once_with(self.servers_mock.get, self.server.id, callback=mock.ANY, **kwargs)
    self.servers_mock.resize.assert_called_with(self.server, self.flavors_get_return_value)