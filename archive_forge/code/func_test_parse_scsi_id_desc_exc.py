from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
def test_parse_scsi_id_desc_exc(self):
    vpd_str = '0083'
    self.assertRaises(exceptions.SCSIPageParsingError, self._diskutils._parse_scsi_page_83, _utils.hex_str_to_byte_array(vpd_str))
    vpd_str = '00FF00240103001060002AC00000000000000EA00000869901140004000003F40115000400000001'
    self.assertRaises(exceptions.SCSIPageParsingError, self._diskutils._parse_scsi_page_83, _utils.hex_str_to_byte_array(vpd_str))
    vpd_str = '008300F40103001060002AC00000000000000EA00000869901140004000003F40115000400000001'
    self.assertRaises(exceptions.SCSIPageParsingError, self._diskutils._parse_scsi_page_83, _utils.hex_str_to_byte_array(vpd_str))
    vpd_str = '00830024010300FF60002AC00000000000000EA00000869901140004000003F40115000400000001'
    self.assertRaises(exceptions.SCSIIdDescriptorParsingError, self._diskutils._parse_scsi_page_83, _utils.hex_str_to_byte_array(vpd_str))
    vpd_str = '0083001F0103001060002AC00000000000000EA00000869901140004000003F4011500'
    self.assertRaises(exceptions.SCSIIdDescriptorParsingError, self._diskutils._parse_scsi_page_83, _utils.hex_str_to_byte_array(vpd_str))