from novaclient.tests.functional.v2.legacy import test_consoles
class TestConsolesNovaClientV28(test_consoles.TestConsolesNovaClient):
    """Consoles functional tests for >=v2.8 api microversions."""
    COMPUTE_API_VERSION = '2.8'

    def test_webmks_console_get(self):
        self._test_console_get('get-mks-console %s ', 'webmks')