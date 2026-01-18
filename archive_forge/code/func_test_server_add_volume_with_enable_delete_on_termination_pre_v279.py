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
@mock.patch.object(sdk_utils, 'supports_microversion')
def test_server_add_volume_with_enable_delete_on_termination_pre_v279(self, sm_mock):

    def side_effect(compute_client, version):
        if version == '2.79':
            return False
        return True
    sm_mock.side_effect = side_effect
    arglist = [self.servers[0].id, self.volumes[0].id, '--enable-delete-on-termination']
    verifylist = [('server', self.servers[0].id), ('volume', self.volumes[0].id), ('enable_delete_on_termination', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-compute-api-version 2.79 or greater is required', str(ex))