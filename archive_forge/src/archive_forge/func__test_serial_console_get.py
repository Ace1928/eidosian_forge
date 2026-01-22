from tempest.lib import exceptions
from novaclient.tests.functional import base
def _test_serial_console_get(self):
    self._test_console_get('get-serial-console %s', 'serial')