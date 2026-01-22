import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
class AsParamPropertyWrapper:

    def __init__(self, param):
        self._param = param

    def getParameter(self):
        return self._param
    _as_parameter_ = property(getParameter)