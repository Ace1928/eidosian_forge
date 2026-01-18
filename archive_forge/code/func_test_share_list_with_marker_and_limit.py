import argparse
import ddt
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.api_versions import MAX_VERSION
from manilaclient.common.apiclient import exceptions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share as osc_shares
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_list_with_marker_and_limit(self):
    arglist = ['--marker', self.new_share.id, '--limit', '2']
    verifylist = [('long', False), ('all_projects', False), ('name', None), ('status', None), ('marker', self.new_share.id), ('limit', 2)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    cmd_columns, cmd_data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, cmd_columns)
    search_opts = self._get_search_opts()
    search_opts['limit'] = 2
    search_opts['offset'] = self.new_share.id
    data = self._get_data()
    self.shares_mock.list.assert_called_once_with(search_opts=search_opts)
    self.assertEqual(data, tuple(cmd_data))