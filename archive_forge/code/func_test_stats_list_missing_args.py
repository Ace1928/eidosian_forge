from unittest import mock
from magnumclient.osc.v1 import stats as osc_stats
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_stats_list_missing_args(self):
    arglist = []
    verifylist = []
    self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)