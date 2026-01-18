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
@mock.patch.object(iso8601, 'parse_date', side_effect=iso8601.ParseError)
def test_server_list_with_invalid_changes_before(self, mock_parse_isotime):
    self._set_mock_microversion('2.66')
    arglist = ['--changes-before', 'Invalid time value']
    verifylist = [('changes_before', 'Invalid time value')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('Invalid changes-before value: Invalid time value', str(e))
    mock_parse_isotime.assert_called_once_with('Invalid time value')