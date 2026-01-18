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
def test_share_delete_wrong_name(self):
    shares = self.setup_shares_mock(count=1)
    arglist = [shares[0].name + '-wrong-name']
    verifylist = [('force', False), ('share_group', None), ('shares', [shares[0].name + '-wrong-name'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertIsNone(result)
    self.shares_mock.delete.side_effect = exceptions.CommandError()
    self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)