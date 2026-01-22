from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
@mock.patch('time.sleep')
def _check_modify_virt_resource_max_retries(self, mock_sleep, side_effect, num_calls=1, expected_fail=False):
    mock_svc = mock.MagicMock()
    self.jobutils._vs_man_svc_attr = mock_svc
    mock_svc.ModifyResourceSettings.side_effect = side_effect
    mock_res_setting_data = mock.MagicMock()
    mock_res_setting_data.GetText_.return_value = mock.sentinel.res_data
    if expected_fail:
        self.assertRaises(exceptions.HyperVException, self.jobutils.modify_virt_resource, mock_res_setting_data)
    else:
        self.jobutils.modify_virt_resource(mock_res_setting_data)
    mock_calls = [mock.call(ResourceSettings=[mock.sentinel.res_data])] * num_calls
    mock_svc.ModifyResourceSettings.assert_has_calls(mock_calls)
    if num_calls > 1:
        mock_sleep.assert_has_calls([mock.call(1)] * (num_calls - 1))
    else:
        mock_sleep.assert_not_called()