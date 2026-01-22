import http.client
from keystone.tests.unit import test_v3
class TestSimpleCert(BaseTestCase):

    def request_cert(self, path):
        self.request(app=self.public_app, method='GET', path=path, expected_status=http.client.GONE)

    def test_ca_cert(self):
        self.request_cert(self.CA_PATH)

    def test_signing_cert(self):
        self.request_cert(self.CERT_PATH)