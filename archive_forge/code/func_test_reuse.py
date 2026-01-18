import contextlib
import gc
import pickle
import runpy
import subprocess
import sys
import unittest
from multiprocessing import get_context
import numba
from numba.core.errors import TypingError
from numba.tests.support import TestCase
from numba.core.target_extension import resolve_dispatcher_from_str
from numba.cloudpickle import dumps, loads
def test_reuse(self):
    """
        Check that deserializing the same function multiple times re-uses
        the same dispatcher object.

        Note that "same function" is intentionally under-specified.
        """
    from .serialize_usecases import closure
    func = closure(5)
    pickled = pickle.dumps(func)
    func2 = closure(6)
    pickled2 = pickle.dumps(func2)
    f = pickle.loads(pickled)
    g = pickle.loads(pickled)
    h = pickle.loads(pickled2)
    self.assertIs(f, g)
    self.assertEqual(f(2, 3), 10)
    g.disable_compile()
    self.assertEqual(g(2, 4), 11)
    self.assertIsNot(f, h)
    self.assertEqual(h(2, 3), 11)
    func = closure(7)
    func(42, 43)
    pickled = pickle.dumps(func)
    del func
    gc.collect()
    f = pickle.loads(pickled)
    g = pickle.loads(pickled)
    self.assertIs(f, g)
    self.assertEqual(f(2, 3), 12)
    g.disable_compile()
    self.assertEqual(g(2, 4), 13)