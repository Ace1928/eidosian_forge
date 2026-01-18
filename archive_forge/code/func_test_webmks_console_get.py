from novaclient.tests.functional.v2.legacy import test_consoles
def test_webmks_console_get(self):
    self._test_console_get('get-mks-console %s ', 'webmks')