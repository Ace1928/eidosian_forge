import typing as py_typing
import unittest
from numba.core import types
from numba.core.errors import TypingError
from numba.core.typing.typeof import typeof
from numba.core.typing.asnumbatype import as_numba_type, AsNumbaTypeRegistry
from numba.experimental.jitclass import jitclass
from numba.tests.support import TestCase
def test_single_containers(self):
    self.assertEqual(as_numba_type(py_typing.List[float]), types.ListType(self.float_nb_type))
    self.assertEqual(as_numba_type(py_typing.Dict[float, str]), types.DictType(self.float_nb_type, self.str_nb_type))
    self.assertEqual(as_numba_type(py_typing.Set[complex]), types.Set(self.complex_nb_type))
    self.assertEqual(as_numba_type(py_typing.Tuple[float, float]), types.Tuple([self.float_nb_type, self.float_nb_type]))
    self.assertEqual(as_numba_type(py_typing.Tuple[float, complex]), types.Tuple([self.float_nb_type, self.complex_nb_type]))