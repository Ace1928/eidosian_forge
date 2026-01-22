import logging
import uuid
from osc_lib import exceptions
from osc_lib import utils as oscutils
from unittest import mock
from manilaclient import api_versions
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestDeleteShareGroupSnapshot(TestShareGroupSnapshot):

    def setUp(self):
        super(TestDeleteShareGroupSnapshot, self).setUp()
        self.share_group_snapshot = manila_fakes.FakeShareGroupSnapshot.create_one_share_group_snapshot()
        self.group_snapshot_mocks.get.return_value = self.share_group_snapshot
        self.cmd = osc_share_group_snapshots.DeleteShareGroupSnapshot(self.app, None)

    def test_share_group_snapshot_delete_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_group_snapshot_delete(self):
        arglist = [self.share_group_snapshot.id]
        verifylist = [('share_group_snapshot', [self.share_group_snapshot.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.group_snapshot_mocks.delete.assert_called_with(self.share_group_snapshot, force=False)
        self.assertIsNone(result)

    def test_share_group_snapshot_delete_force(self):
        arglist = [self.share_group_snapshot.id, '--force']
        verifylist = [('share_group_snapshot', [self.share_group_snapshot.id]), ('force', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.group_snapshot_mocks.delete.assert_called_with(self.share_group_snapshot, force=True)
        self.assertIsNone(result)

    def test_share_group_snapshot_delete_multiple(self):
        share_group_snapshots = manila_fakes.FakeShareGroupSnapshot.create_share_group_snapshots(count=2)
        arglist = [share_group_snapshots[0].id, share_group_snapshots[1].id]
        verifylist = [('share_group_snapshot', [share_group_snapshots[0].id, share_group_snapshots[1].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertEqual(self.group_snapshot_mocks.delete.call_count, len(share_group_snapshots))
        self.assertIsNone(result)

    def test_share_group_snapshot_delete_exception(self):
        arglist = [self.share_group_snapshot.id]
        verifylist = [('share_group_snapshot', [self.share_group_snapshot.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.group_snapshot_mocks.delete.side_effect = exceptions.CommandError()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_group_snapshot_delete_wait(self):
        arglist = [self.share_group_snapshot.id, '--wait']
        verifylist = [('share_group_snapshot', [self.share_group_snapshot.id]), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=True):
            result = self.cmd.take_action(parsed_args)
            self.group_snapshot_mocks.delete.assert_called_with(self.share_group_snapshot, force=False)
            self.group_snapshot_mocks.get.assert_called_with(self.share_group_snapshot.id)
            self.assertIsNone(result)

    def test_share_group_snapshot_delete_wait_exception(self):
        arglist = [self.share_group_snapshot.id, '--wait']
        verifylist = [('share_group_snapshot', [self.share_group_snapshot.id]), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=False):
            self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)