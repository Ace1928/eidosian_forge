from oslo_middleware import opts
from oslotest.base import BaseTestCase
class TestOptionDiscovery(BaseTestCase):

    def test_all(self):
        opts.list_opts()

    def test_sizelimit(self):
        opts.list_opts_sizelimit()

    def test_cors(self):
        opts.list_opts_cors()