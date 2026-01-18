from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share_snapshots as osc_share_snapshots
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@mock.patch('manilaclient.osc.v2.share_snapshots.LOG')
def test_share_snapshot_create_wait_error(self, mock_logger):
    arglist = [self.share.id, '--wait']
    verifylist = [('share', self.share.id), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch('osc_lib.utils.wait_for_status', return_value=False):
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.create.assert_called_with(share=self.share, force=False, name=None, description=None, metadata={})
        mock_logger.error.assert_called_with('ERROR: Share snapshot is in error state.')
        self.snapshots_mock.get.assert_called_with(self.share_snapshot.id)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)