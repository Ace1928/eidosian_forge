from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
@mock.patch.object(_utils, 'get_com_error_hresult')
def test_not_found_decorator(self, mock_get_com_error_hresult):
    mock_get_com_error_hresult.side_effect = lambda x: x
    translated_exc = exceptions.HyperVVMNotFoundException

    @_utils.not_found_decorator(translated_exc=translated_exc)
    def f(to_call):
        to_call()
    to_call = mock.Mock()
    to_call.side_effect = exceptions.x_wmi('expected error', com_error=_utils._WBEM_E_NOT_FOUND)
    self.assertRaises(translated_exc, f, to_call)
    to_call.side_effect = exceptions.x_wmi()
    self.assertRaises(exceptions.x_wmi, f, to_call)