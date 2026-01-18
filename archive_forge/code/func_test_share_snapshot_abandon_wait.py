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
def test_share_snapshot_abandon_wait(self):
    arglist = [self.share_snapshot.id, '--wait']
    verifylist = [('snapshot', [self.share_snapshot.id]), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch('osc_lib.utils.wait_for_delete', return_value=True):
        result = self.cmd.take_action(parsed_args)
        self.snapshots_mock.unmanage.assert_called_with(self.share_snapshot)
        self.assertIsNone(result)