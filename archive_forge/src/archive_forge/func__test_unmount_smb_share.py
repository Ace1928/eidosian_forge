from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import smbutils
def _test_unmount_smb_share(self, force=False):
    fake_mapping = mock.Mock()
    fake_mapping_attr_err = mock.Mock()
    fake_mapping_attr_err.side_effect = AttributeError
    smb_mapping_class = self._smb_conn.Msft_SmbMapping
    smb_mapping_class.return_value = [fake_mapping, fake_mapping_attr_err]
    self._smbutils.unmount_smb_share(mock.sentinel.share_path, force)
    smb_mapping_class.assert_called_once_with(RemotePath=mock.sentinel.share_path)
    fake_mapping.Remove.assert_called_once_with(Force=force)