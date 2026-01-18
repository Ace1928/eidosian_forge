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
def test_unshelve_with_no_az_and_az_conflict(self):
    self._set_mock_microversion('2.91')
    arglist = ['--availability-zone', 'foo-az', '--no-availability-zone', self.server.id]
    verifylist = [('availability_zone', 'foo-az'), ('no_availability_zone', True), ('server', [self.server.id])]
    ex = self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
    self.assertIn('argument --no-availability-zone: not allowed with argument --availability-zone', str(ex))