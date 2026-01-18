import sys
import subprocess
from itertools import product
from textwrap import dedent
import numpy as np
from numba import config
from numba import njit
from numba import int32, float32, prange, uint8
from numba.core import types
from numba import typeof
from numba.typed import List, Dict
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.core.unsafe.refcount import get_refcount
from numba.experimental import jitclass
def test_sort_dispatcher_key(self):

    def udt(lst, key):
        lst.sort(key=key)
        return lst
    my_lists = self.make_both(np.random.randint(0, 100, 31))
    py_key = lambda x: x + 1
    nb_key = njit(lambda x: x + 1)
    self.assertEqual(list(udt(my_lists['nb'], key=nb_key)), udt(my_lists['py'], key=py_key))
    self.assertEqual(list(udt(my_lists['nb'], key=nb_key)), list(udt(my_lists['nb'], key=py_key)))