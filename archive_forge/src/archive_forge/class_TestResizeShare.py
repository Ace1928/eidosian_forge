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
class TestResizeShare(TestShare):

    def setUp(self):
        super(TestResizeShare, self).setUp()
        self._share = manila_fakes.FakeShare.create_one_share(attrs={'size': 2, 'status': 'available'})
        self.shares_mock.get.return_value = self._share
        self.cmd = osc_shares.ResizeShare(self.app, None)

    def test_share_shrink(self):
        arglist = [self._share.id, '1']
        verifylist = [('share', self._share.id), ('new_size', 1)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.shares_mock.shrink.assert_called_with(self._share, 1)

    def test_share_shrink_exception(self):
        arglist = [self._share.id, '1']
        verifylist = [('share', self._share.id), ('new_size', 1)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.shares_mock.shrink.side_effect = Exception()
        self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_extend(self):
        arglist = [self._share.id, '3']
        verifylist = [('share', self._share.id), ('new_size', 3)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.shares_mock.extend.assert_called_with(self._share, 3)

    def test_share_extend_exception(self):
        arglist = [self._share.id, '3']
        verifylist = [('share', self._share.id), ('new_size', 3)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.shares_mock.extend.side_effect = Exception()
        self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_resize_already_required_size(self):
        arglist = [self._share.id, '2']
        verifylist = [('share', self._share.id), ('new_size', 2)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_missing_args(self):
        arglist = [self._share.id]
        verifylist = [('share', self._share.id)]
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_resize_wait(self):
        arglist = [self._share.id, '3', '--wait']
        verifylist = [('share', self._share.id), ('new_size', 3), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.shares_mock.extend.assert_called_with(self._share, 3)
        self.shares_mock.get.assert_called_with(self._share.id)

    def test_share_resize_wait_error(self):
        arglist = [self._share.id, '3', '--wait']
        verifylist = [('share', self._share.id), ('new_size', 3), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_status', return_value=False):
            self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)