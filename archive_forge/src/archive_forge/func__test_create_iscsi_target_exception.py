from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
@mock.patch('os_win._utils.get_com_error_code')
def _test_create_iscsi_target_exception(self, mock_get_com_err_code, target_exists=False, fail_if_exists=False):
    mock_wt_host_cls = self._tgutils._conn_wmi.WT_Host
    mock_wt_host_cls.NewHost.side_effect = test_base.FakeWMIExc
    mock_get_com_err_code.return_value = self._tgutils._ERR_FILE_EXISTS if target_exists else 1
    if target_exists and (not fail_if_exists):
        self._tgutils.create_iscsi_target(mock.sentinel.target_name, fail_if_exists=fail_if_exists)
    else:
        self.assertRaises(exceptions.ISCSITargetException, self._tgutils.create_iscsi_target, mock.sentinel.target_name, fail_if_exists=fail_if_exists)
    mock_wt_host_cls.NewHost.assert_called_once_with(HostName=mock.sentinel.target_name)