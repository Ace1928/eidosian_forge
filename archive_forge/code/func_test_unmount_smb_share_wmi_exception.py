from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import smbutils
def test_unmount_smb_share_wmi_exception(self):
    fake_mapping = mock.Mock()
    fake_mapping.Remove.side_effect = exceptions.x_wmi
    self._smb_conn.Msft_SmbMapping.return_value = [fake_mapping]
    self.assertRaises(exceptions.SMBException, self._smbutils.unmount_smb_share, mock.sentinel.share_path, force=True)