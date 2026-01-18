from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
def test_select_supported_scsi_identifiers(self):
    identifiers = [{'type': id_type} for id_type in constants.SUPPORTED_SCSI_UID_FORMATS[::-1]]
    identifiers.append({'type': mock.sentinel.scsi_id_format})
    expected_identifiers = [{'type': id_type} for id_type in constants.SUPPORTED_SCSI_UID_FORMATS]
    result = self._diskutils._select_supported_scsi_identifiers(identifiers)
    self.assertEqual(expected_identifiers, result)