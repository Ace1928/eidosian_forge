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
@mock.patch.object(getpass, 'getpass', return_value=mock.sentinel.fake_pass)
def test_server_set_with_root_password(self, mock_getpass):
    arglist = ['--root-password', 'foo_vm']
    verifylist = [('root_password', True), ('server', 'foo_vm')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.fake_servers[0].change_password.assert_called_once_with(mock.sentinel.fake_pass)
    self.assertIsNone(result)