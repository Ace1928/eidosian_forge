import logging
import uuid
from osc_lib import exceptions
from osc_lib import utils as oscutils
from unittest import mock
from manilaclient import api_versions
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_group_snapshot_list_no_options(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.group_snapshot_mocks.list.assert_called_once_with(search_opts={'all_tenants': False, 'name': None, 'status': None, 'share_group_id': None, 'limit': None, 'offset': None})
    self.assertEqual(self.columns, columns)
    self.assertEqual(list(self.values), list(data))