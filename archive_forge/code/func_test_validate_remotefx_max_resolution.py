from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_validate_remotefx_max_resolution(self):
    self.assertRaises(exceptions.HyperVRemoteFXException, self._vmutils._validate_remotefx_params, 1, '1024x700')