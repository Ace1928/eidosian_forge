from unittest import mock
from magnumclient.osc.v1 import certificates as osc_certificates
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_sign_ca_without_csr(self):
    arglist = ['fake-cluster']
    verifylist = [('cluster', 'fake-cluster')]
    self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)