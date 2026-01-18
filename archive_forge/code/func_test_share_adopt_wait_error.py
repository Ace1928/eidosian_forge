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
@mock.patch('manilaclient.osc.v2.share.LOG')
def test_share_adopt_wait_error(self, mock_logger):
    arglist = ['some.host@driver#pool', 'NFS', '10.0.0.1:/example_path', '--wait']
    verifylist = [('service_host', 'some.host@driver#pool'), ('protocol', 'NFS'), ('export_path', '10.0.0.1:/example_path'), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch('osc_lib.utils.wait_for_status', return_value=False):
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.manage.assert_called_with(description=None, export_path='10.0.0.1:/example_path', name=None, protocol='NFS', service_host='some.host@driver#pool')
        mock_logger.error.assert_called_with('ERROR: Share is in error state.')
        self.shares_mock.get.assert_called_with(self._share.id)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)