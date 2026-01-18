from unittest import mock
from magnumclient.osc.v1 import certificates as osc_certificates
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_sign_ca_without_cluster(self):
    arglist = [self.test_csr_path]
    verifylist = [('csr', self.test_csr_path)]
    self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)