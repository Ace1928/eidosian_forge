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
def test_unset_snapshot_name_exception(self):
    arglist = [self.share_snapshot.id, '--name']
    verifylist = [('snapshot', self.share_snapshot.id), ('name', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.snapshots_mock.update.side_effect = Exception()
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)