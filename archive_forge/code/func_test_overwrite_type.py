import typing as py_typing
import unittest
from numba.core import types
from numba.core.errors import TypingError
from numba.core.typing.typeof import typeof
from numba.core.typing.asnumbatype import as_numba_type, AsNumbaTypeRegistry
from numba.experimental.jitclass import jitclass
from numba.tests.support import TestCase
def test_overwrite_type(self):
    as_numba_type = AsNumbaTypeRegistry()
    self.assertEqual(as_numba_type(float), self.float_nb_type)
    as_numba_type.register(float, types.float32)
    self.assertEqual(as_numba_type(float), types.float32)
    self.assertNotEqual(as_numba_type(float), self.float_nb_type)