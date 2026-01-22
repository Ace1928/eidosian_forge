from unittest import mock
from magnumclient.osc.v1 import certificates as osc_certificates
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
class TestCertificate(magnum_fakes.TestMagnumClientOSCV1):

    def setUp(self):
        super(TestCertificate, self).setUp()
        self.clusters_mock = self.app.client_manager.container_infra.clusters