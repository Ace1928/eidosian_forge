import unittest
from ctypes import *
import _ctypes_test
def wndproc(hwnd, msg, wParam, lParam):
    return hwnd + msg + wParam + lParam