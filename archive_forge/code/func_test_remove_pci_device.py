from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_remove_pci_device(self):
    self.assertRaises(NotImplementedError, self._vmutils.remove_pci_device, mock.sentinel.vm_name, mock.sentinel.vendor_id, mock.sentinel.product_id)