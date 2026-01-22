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
class TestShareRestore(TestShare):

    def setUp(self):
        super(TestShareRestore, self).setUp()
        self.share = manila_fakes.FakeShare.create_one_share(methods={'restore': None})
        self.shares_mock.get.return_value = self.share
        self.cmd = osc_shares.RestoreShare(self.app, None)

        def test_share_restore(self):
            arglist = [self.share.name]
            verifylist = [('share', [self.share.name])]
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            result = self.cmd.take_action(parsed_args)
            self.shares_mock.get.assert_called_with(self.share)
            self.share.restore.assert_called_with(self.share)
            self.assertIsNone(result)

        def test_share_restore_exception(self):
            arglist = [self.share.name]
            verifylist = [('share', [self.share.name])]
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            self.share.restore.side_effect = Exception()
            self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)