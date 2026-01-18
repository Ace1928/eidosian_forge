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
def test_share_snapshot_access_deny_multiple(self):
    access_rules = manila_fakes.FakeSnapshotAccessRule.create_access_rules(count=2)
    arglist = [self.share_snapshot.id, access_rules[0].id, access_rules[1].id]
    verifylist = [('snapshot', self.share_snapshot.id), ('id', [access_rules[0].id, access_rules[1].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertEqual(self.snapshots_mock.deny.call_count, len(access_rules))
    self.assertIsNone(result)