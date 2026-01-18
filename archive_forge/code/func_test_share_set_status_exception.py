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
def test_share_set_status_exception(self):
    new_status = 'available'
    arglist = [self._share.id, '--status', new_status]
    verifylist = [('share', self._share.id), ('status', new_status)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self._share.reset_state.side_effect = Exception()
    self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)