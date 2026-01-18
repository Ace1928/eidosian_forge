from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_set_secure_boot(self):
    vs_data = mock.MagicMock()
    self._vmutils._set_secure_boot(vs_data, msft_ca_required=False)
    self.assertTrue(vs_data.SecureBootEnabled)