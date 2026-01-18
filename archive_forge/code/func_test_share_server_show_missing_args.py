from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_servers as osc_share_servers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_server_show_missing_args(self):
    arglist = []
    verifylist = []
    self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)