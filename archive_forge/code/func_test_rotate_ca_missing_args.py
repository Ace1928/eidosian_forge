from unittest import mock
from magnumclient.osc.v1 import certificates as osc_certificates
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_rotate_ca_missing_args(self):
    arglist = []
    verifylist = []
    self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)