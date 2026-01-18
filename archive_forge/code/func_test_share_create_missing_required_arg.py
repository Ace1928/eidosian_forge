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
def test_share_create_missing_required_arg(self):
    """Verifies missing required arguments."""
    arglist = [self.new_share.share_proto]
    verifylist = [('share_proto', self.new_share.share_proto)]
    self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)