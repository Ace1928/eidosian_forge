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
def test_server_add_volume_with_disable_and_enable_delete_on_termination(self, sm_mock):
    arglist = ['--enable-delete-on-termination', '--disable-delete-on-termination', '--device', '/dev/sdb', self.servers[0].id, self.volumes[0].id]
    verifylist = [('server', self.servers[0].id), ('volume', self.volumes[0].id), ('device', '/dev/sdb'), ('enable_delete_on_termination', True), ('disable_delete_on_termination', True)]
    ex = self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
    self.assertIn('argument --disable-delete-on-termination: not allowed with argument --enable-delete-on-termination', str(ex))