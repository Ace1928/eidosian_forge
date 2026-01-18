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
def test_server_add_floating_ip_fixed(self, fip_mock):
    _floating_ip = compute_fakes.create_one_floating_ip()
    arglist = ['--fixed-ip-address', _floating_ip['fixed_ip'], 'server1', _floating_ip['ip']]
    verifylist = [('fixed_ip_address', _floating_ip['fixed_ip']), ('server', 'server1'), ('ip_address', _floating_ip['ip'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    fip_mock.assert_called_once_with('server1', _floating_ip['ip'], fixed_address=_floating_ip['fixed_ip'])