from unittest import mock
import ddt
from os_brick import exception
from os_brick.remotefs import windows_remotefs
from os_brick.tests import base
def test_unmount(self):
    self._remotefs.unmount(self._FAKE_SHARE)
    self._smbutils.unmount_smb_share.assert_called_once_with(self._FAKE_SHARE)