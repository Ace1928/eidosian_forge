from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
@mock.patch.object(_utils, 'time')
def test_retry_decorator(self, mock_time):
    err_code = 1
    max_retry_count = 5
    max_sleep_time = 2
    timeout = max_retry_count + 1
    mock_time.time.side_effect = range(timeout)
    raised_exc = exceptions.Win32Exception(message='fake_exc', error_code=err_code)
    side_effect = [raised_exc] * max_retry_count
    side_effect.append(mock.sentinel.ret_val)
    fake_func, fake_func_side_effect = self._get_fake_func_with_retry_decorator(error_codes=err_code, exceptions=exceptions.Win32Exception, max_retry_count=max_retry_count, max_sleep_time=max_sleep_time, timeout=timeout, side_effect=side_effect)
    ret_val = fake_func(mock.sentinel.arg, kwarg=mock.sentinel.kwarg)
    self.assertEqual(mock.sentinel.ret_val, ret_val)
    fake_func_side_effect.assert_has_calls([mock.call(mock.sentinel.arg, kwarg=mock.sentinel.kwarg)] * (max_retry_count + 1))
    self.assertEqual(max_retry_count + 1, mock_time.time.call_count)
    mock_time.sleep.assert_has_calls([mock.call(sleep_time) for sleep_time in [1, 2, 2, 2, 1]])