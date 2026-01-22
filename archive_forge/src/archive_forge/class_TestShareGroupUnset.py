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
class TestShareGroupUnset(TestShareGroup):

    def setUp(self):
        super(TestShareGroupUnset, self).setUp()
        self.share_group = manila_fakes.FakeShareGroup.create_one_share_group()
        self.groups_mock.get.return_value = self.share_group
        self.cmd = osc_share_groups.UnsetShareGroup(self.app, None)

    def test_unset_share_group_name(self):
        arglist = [self.share_group.id, '--name']
        verifylist = [('share_group', self.share_group.id), ('name', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.groups_mock.update.assert_called_with(self.share_group, name=None)
        self.assertIsNone(result)

    def test_unset_share_group_description(self):
        arglist = [self.share_group.id, '--description']
        verifylist = [('share_group', self.share_group.id), ('description', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.groups_mock.update.assert_called_with(self.share_group, description=None)
        self.assertIsNone(result)

    def test_unset_share_group_name_exception(self):
        arglist = [self.share_group.id, '--name']
        verifylist = [('share_group', self.share_group.id), ('name', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.groups_mock.update.side_effect = Exception()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)