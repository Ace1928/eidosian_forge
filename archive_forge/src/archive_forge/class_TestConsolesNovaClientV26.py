from novaclient.tests.functional.v2.legacy import test_consoles
class TestConsolesNovaClientV26(test_consoles.TestConsolesNovaClient):
    """Consoles functional tests for >=v2.6 api microversions."""
    COMPUTE_API_VERSION = '2.6'

    def test_vnc_console_get(self):
        self._test_vnc_console_get()

    def test_spice_console_get(self):
        self._test_spice_console_get()

    def test_rdp_console_get(self):
        self._test_rdp_console_get()

    def test_serial_console_get(self):
        self._test_serial_console_get()