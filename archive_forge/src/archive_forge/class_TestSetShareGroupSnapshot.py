import logging
import uuid
from osc_lib import exceptions
from osc_lib import utils as oscutils
from unittest import mock
from manilaclient import api_versions
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestSetShareGroupSnapshot(TestShareGroupSnapshot):

    def setUp(self):
        super(TestSetShareGroupSnapshot, self).setUp()
        self.share_group_snapshot = manila_fakes.FakeShareGroupSnapshot.create_one_share_group_snapshot()
        self.group_snapshot_mocks.get.return_value = self.share_group_snapshot
        self.cmd = osc_share_group_snapshots.SetShareGroupSnapshot(self.app, None)
        self.data = tuple(self.share_group_snapshot._info.values())
        self.columns = tuple(self.share_group_snapshot._info.keys())

    def test_set_share_group_snapshot_name_description(self):
        group_snapshot_name = 'group-snapshot-name-' + uuid.uuid4().hex
        group_snapshot_description = 'group-snapshot-description-' + uuid.uuid4().hex
        arglist = [self.share_group_snapshot.id, '--name', group_snapshot_name, '--description', group_snapshot_description]
        verifylist = [('share_group_snapshot', self.share_group_snapshot.id), ('name', group_snapshot_name), ('description', group_snapshot_description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.group_snapshot_mocks.update.assert_called_with(self.share_group_snapshot, name=parsed_args.name, description=parsed_args.description)
        self.assertIsNone(result)

    def test_set_share_group_snapshot_status(self):
        arglist = [self.share_group_snapshot.id, '--status', 'available']
        verifylist = [('share_group_snapshot', self.share_group_snapshot.id), ('status', 'available')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.group_snapshot_mocks.reset_state.assert_called_with(self.share_group_snapshot, 'available')
        self.assertIsNone(result)

    def test_set_share_group_snapshot_exception(self):
        arglist = [self.share_group_snapshot.id, '--status', 'available']
        verifylist = [('share_group_snapshot', self.share_group_snapshot.id), ('status', 'available')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.group_snapshot_mocks.reset_state.side_effect = Exception()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)