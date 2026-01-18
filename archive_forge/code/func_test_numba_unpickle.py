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
def test_numba_unpickle(self):
    from numba.core.serialize import _numba_unpickle
    random_obj = object()
    bytebuf = pickle.dumps(random_obj)
    hashed = hash(random_obj)
    got1 = _numba_unpickle(id(random_obj), bytebuf, hashed)
    self.assertIsNot(got1, random_obj)
    got2 = _numba_unpickle(id(random_obj), bytebuf, hashed)
    self.assertIs(got1, got2)