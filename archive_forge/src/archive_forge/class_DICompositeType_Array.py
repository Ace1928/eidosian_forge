from numba.tests.support import TestCase, linux_only
from numba.tests.gdb_support import needs_gdb, skip_unless_pexpect, GdbMIDriver
from unittest.mock import patch, Mock
from numba.core import datamodel
import numpy as np
from numba import typeof
import ctypes as ct
import unittest
class DICompositeType_Array:

    def __init__(self, arr, type_str):
        self._arr = arr
        self._type_str = type_str

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def data(self):
        return self._arr.ctypes.data

    @property
    def itemsize(self):
        return self._arr.itemsize

    @property
    def shape(self):
        return DIDerivedType_tuple(self._arr.shape)

    @property
    def strides(self):
        return DIDerivedType_tuple(self._arr.strides)

    @property
    def type(self):
        return self._type_str