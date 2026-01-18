from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import smbutils
def test_remake_unavailable_mapping(self):
    self._test_check_smb_mapping(existing_mappings=True, share_available=False)