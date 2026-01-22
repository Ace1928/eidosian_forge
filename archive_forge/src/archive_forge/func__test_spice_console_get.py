from tempest.lib import exceptions
from novaclient.tests.functional import base
def _test_spice_console_get(self):
    self._test_console_get('get-spice-console %s spice-html5', 'spice-html5')