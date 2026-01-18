from collections import namedtuple
import gc
import os
import operator
import sys
import weakref
import numpy as np
from numba.core import types, typing, errors, sigutils
from numba.core.types.abstract import _typecache
from numba.core.types.functions import _header_lead
from numba.core.typing.templates import make_overload_template
from numba import jit, njit, typeof
from numba.core.extending import (overload, register_model, models, unbox,
from numba.tests.support import TestCase, create_temp_module
from numba.tests.enum_usecases import Color, Shake, Shape
import unittest
from numba.np import numpy_support
from numba.core import types
def test_record_type_equiv(self):
    rec_dt = np.dtype([('a', np.int32), ('b', np.float32)])
    rec_ty = typeof(rec_dt)
    art1 = rec_ty[::1]
    arr = np.zeros(5, dtype=rec_dt)
    art2 = typeof(arr)
    self.assertEqual(art2.dtype.dtype, rec_ty)
    self.assertEqual(art1, art2)