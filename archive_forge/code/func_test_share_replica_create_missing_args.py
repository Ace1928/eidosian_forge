from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_replicas as osc_share_replicas
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_replica_create_missing_args(self):
    arglist = []
    verifylist = []
    self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)