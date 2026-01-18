from ctypes import *
import unittest, sys
from test import support
import _ctypes_test
def test_winerror(self):
    import errno
    ERROR_INVALID_PARAMETER = 87
    msg = FormatError(ERROR_INVALID_PARAMETER).strip()
    args = (errno.EINVAL, msg, None, ERROR_INVALID_PARAMETER)
    e = WinError(ERROR_INVALID_PARAMETER)
    self.assertEqual(e.args, args)
    self.assertEqual(e.errno, errno.EINVAL)
    self.assertEqual(e.winerror, ERROR_INVALID_PARAMETER)
    windll.kernel32.SetLastError(ERROR_INVALID_PARAMETER)
    try:
        raise WinError()
    except OSError as exc:
        e = exc
    self.assertEqual(e.args, args)
    self.assertEqual(e.errno, errno.EINVAL)
    self.assertEqual(e.winerror, ERROR_INVALID_PARAMETER)