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
class TestShareDelete(TestShare):

    def setUp(self):
        super(TestShareDelete, self).setUp()
        self.shares_mock.delete = mock.Mock()
        self.shares_mock.delete.return_value = None
        self.cmd = osc_shares.DeleteShare(self.app, None)

    def test_share_delete_one(self):
        shares = self.setup_shares_mock(count=1)
        arglist = [shares[0].name]
        verifylist = [('force', False), ('share_group', None), ('shares', [shares[0].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.shares_mock.delete.assert_called_with(shares[0], None)
        self.shares_mock.soft_delete.assert_not_called()
        self.shares_mock.force_delete.assert_not_called()
        self.assertIsNone(result)

    def test_share_delete_many(self):
        shares = self.setup_shares_mock(count=3)
        arglist = [v.id for v in shares]
        verifylist = [('force', False), ('share_group', None), ('shares', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = [mock.call(s, None) for s in shares]
        self.shares_mock.delete.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_share_delete_with_share_group(self):
        shares = self.setup_shares_mock(count=1)
        share_group = self.setup_share_groups_mock()
        arglist = [shares[0].name, '--share-group', share_group['id']]
        verifylist = [('share_group', share_group['id']), ('shares', [shares[0].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.shares_mock.delete.assert_called_with(shares[0], share_group['id'])
        self.assertIsNone(result)

    def test_share_delete_with_force(self):
        shares = self.setup_shares_mock(count=1)
        arglist = ['--force', shares[0].name]
        verifylist = [('force', True), ('share_group', None), ('shares', [shares[0].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.shares_mock.force_delete.assert_called_once_with(shares[0])
        self.shares_mock.delete.assert_not_called()
        self.shares_mock.soft_delete.assert_not_called()
        self.assertIsNone(result)

    def test_share_delete_with_soft(self):
        shares = self.setup_shares_mock(count=1)
        arglist = ['--soft', shares[0].name]
        verifylist = [('soft', True), ('share_group', None), ('shares', [shares[0].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.shares_mock.soft_delete.assert_called_once_with(shares[0])
        self.shares_mock.delete.assert_not_called()
        self.shares_mock.force_delete.assert_not_called()
        self.assertIsNone(result)

    def test_share_delete_wrong_name(self):
        shares = self.setup_shares_mock(count=1)
        arglist = [shares[0].name + '-wrong-name']
        verifylist = [('force', False), ('share_group', None), ('shares', [shares[0].name + '-wrong-name'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.shares_mock.delete.side_effect = exceptions.CommandError()
        self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_delete_no_name(self):
        arglist = []
        verifylist = [('force', False), ('share_group', None), ('shares', '')]
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_delete_wait(self):
        shares = self.setup_shares_mock(count=1)
        arglist = [shares[0].name, '--wait']
        verifylist = [('force', False), ('share_group', None), ('shares', [shares[0].name]), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=True):
            result = self.cmd.take_action(parsed_args)
            self.shares_mock.delete.assert_called_with(shares[0], None)
            self.shares_mock.get.assert_called_with(shares[0].name)
            self.assertIsNone(result)

    def test_share_delete_wait_error(self):
        shares = self.setup_shares_mock(count=1)
        arglist = [shares[0].name, '--wait']
        verifylist = [('force', False), ('share_group', None), ('shares', [shares[0].name]), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=False):
            self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)