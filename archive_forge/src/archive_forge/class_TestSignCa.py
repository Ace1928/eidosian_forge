from unittest import mock
from magnumclient.osc.v1 import certificates as osc_certificates
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
class TestSignCa(TestCertificate):
    test_csr_path = 'magnumclient/tests/test_csr/test.csr'

    def setUp(self):
        super(TestSignCa, self).setUp()
        attr = dict()
        attr['name'] = 'fake-cluster-1'
        self._cluster = magnum_fakes.FakeCluster.create_one_cluster(attr)
        self.clusters_mock.get = mock.Mock()
        self.clusters_mock.get.return_value = self._cluster
        self.cmd = osc_certificates.SignCa(self.app, None)

    def test_sign_ca(self):
        arglist = ['fake-cluster', self.test_csr_path]
        verifylist = [('cluster', 'fake-cluster'), ('csr', self.test_csr_path)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.clusters_mock.get.assert_called_once_with('fake-cluster')

    def test_sign_ca_without_csr(self):
        arglist = ['fake-cluster']
        verifylist = [('cluster', 'fake-cluster')]
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)

    def test_sign_ca_without_cluster(self):
        arglist = [self.test_csr_path]
        verifylist = [('csr', self.test_csr_path)]
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show_ca_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)