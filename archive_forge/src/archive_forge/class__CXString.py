from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class _CXString(Structure):
    """Helper for transforming CXString results."""
    _fields_ = [('spelling', c_char_p), ('free', c_int)]

    def __del__(self):
        conf.lib.clang_disposeString(self)

    @staticmethod
    def from_result(res, fn=None, args=None):
        assert isinstance(res, _CXString)
        return conf.lib.clang_getCString(res)