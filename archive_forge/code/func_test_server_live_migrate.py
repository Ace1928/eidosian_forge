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
def test_server_live_migrate(self):
    arglist = ['--live-migration', self.server.id]
    verifylist = [('live_migration', True), ('host', None), ('block_migration', None), ('disk_overcommit', None), ('wait', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.servers_mock.get.assert_called_with(self.server.id)
    self.server.live_migrate.assert_called_with(block_migration=False, disk_over_commit=False, host=None)
    self.assertNotCalled(self.servers_mock.migrate)
    self.assertIsNone(result)