from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import smbutils
def test_soft_unmount_smb_share(self):
    self._test_unmount_smb_share()