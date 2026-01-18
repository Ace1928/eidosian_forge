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
def test_server_list_v269_with_partial_constructs(self):
    self._set_mock_microversion('2.69')
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    server_dict = {'id': 'server-id-95a56bfc4xxxxxx28d7e418bfd97813a', 'status': 'UNKNOWN', 'tenant_id': '6f70656e737461636b20342065766572', 'created': '2018-12-03T21:06:18Z', 'links': [{'href': 'http://fake/v2.1/', 'rel': 'self'}, {'href': 'http://fake', 'rel': 'bookmark'}], 'networks': {}}
    fake_server = compute_fakes.fakes.FakeResource(info=server_dict)
    self.servers.append(fake_server)
    columns, data = self.cmd.take_action(parsed_args)
    next(data)
    next(data)
    next(data)
    partial_server = next(data)
    expected_row = ('server-id-95a56bfc4xxxxxx28d7e418bfd97813a', '', 'UNKNOWN', server.AddressesColumn(''), '', '')
    self.assertEqual(expected_row, partial_server)