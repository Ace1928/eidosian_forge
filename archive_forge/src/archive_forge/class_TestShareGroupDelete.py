import argparse
from unittest import mock
import uuid
from osc_lib import exceptions
from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as oscutils
from manilaclient.osc import utils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_groups as osc_share_groups
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareGroupDelete(TestShareGroup):

    def setUp(self):
        super(TestShareGroupDelete, self).setUp()
        self.share_group = manila_fakes.FakeShareGroup.create_one_share_group()
        self.groups_mock.get.return_value = self.share_group
        self.cmd = osc_share_groups.DeleteShareGroup(self.app, None)

    def test_share_group_delete(self):
        arglist = [self.share_group.id]
        verifylist = [('share_group', [self.share_group.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.groups_mock.delete.assert_called_with(self.share_group, force=False)
        self.assertIsNone(result)

    def test_share_group_delete_force(self):
        arglist = [self.share_group.id, '--force']
        verifylist = [('share_group', [self.share_group.id]), ('force', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.groups_mock.delete.assert_called_with(self.share_group, force=True)
        self.assertIsNone(result)

    def test_share_group_delete_multiple(self):
        share_groups = manila_fakes.FakeShareGroup.create_share_groups(count=2)
        arglist = [share_groups[0].id, share_groups[1].id]
        verifylist = [('share_group', [share_groups[0].id, share_groups[1].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertEqual(self.groups_mock.delete.call_count, len(share_groups))
        self.assertIsNone(result)

    def test_share_group_delete_exception(self):
        arglist = [self.share_group.id]
        verifylist = [('share_group', [self.share_group.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.groups_mock.delete.side_effect = exceptions.CommandError()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_group_delete_wait(self):
        arglist = [self.share_group.id, '--wait']
        verifylist = [('share_group', [self.share_group.id]), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=True):
            result = self.cmd.take_action(parsed_args)
            self.groups_mock.delete.assert_called_with(self.share_group, force=False)
            self.groups_mock.get.assert_called_with(self.share_group.id)
            self.assertIsNone(result)

    def test_share_group_delete_wait_exception(self):
        arglist = [self.share_group.id, '--wait']
        verifylist = [('share_group', [self.share_group.id]), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=False):
            self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)