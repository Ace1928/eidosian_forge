from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
def test_parse_scsi_page_83_no_desc(self):
    vpd_str = '008300000103001060002AC00000000000000EA00000869901140004000003F40115000400000001'
    buff = _utils.hex_str_to_byte_array(vpd_str)
    identifiers = self._diskutils._parse_scsi_page_83(buff)
    self.assertEqual([], identifiers)