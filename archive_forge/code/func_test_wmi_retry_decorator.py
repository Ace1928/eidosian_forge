from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
@mock.patch.object(_utils, 'time')
@ddt.data(True, False)
def test_wmi_retry_decorator(self, expect_hres, mock_time):
    expected_hres = 2147991279
    expected_err_code = expected_hres if expect_hres else 48879
    other_hres = 2147942401
    max_retry_count = 5
    expected_try_count = 2
    side_effect = [test_base.FakeWMIExc(hresult=expected_hres), test_base.FakeWMIExc(hresult=other_hres)]
    decorator = _utils.wmi_retry_decorator_hresult if expect_hres else _utils.wmi_retry_decorator
    fake_func, fake_func_side_effect = self._get_fake_func_with_retry_decorator(error_codes=expected_err_code, max_retry_count=max_retry_count, decorator=decorator, side_effect=side_effect)
    self.assertRaises(test_base.FakeWMIExc, fake_func, mock.sentinel.arg, kwarg=mock.sentinel.kwarg)
    fake_func_side_effect.assert_has_calls([mock.call(mock.sentinel.arg, kwarg=mock.sentinel.kwarg)] * expected_try_count)