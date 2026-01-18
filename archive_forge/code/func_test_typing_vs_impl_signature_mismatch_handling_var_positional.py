import inspect
import math
import operator
import sys
import pickle
import multiprocessing
import ctypes
import warnings
import re
import numpy as np
from llvmlite import ir
import numba
from numba import njit, jit, vectorize, guvectorize, objmode
from numba.core import types, errors, typing, compiler, cgutils
from numba.core.typed_passes import type_inference_stage
from numba.core.registry import cpu_target
from numba.core.imputils import lower_constant
from numba.tests.support import (
from numba.core.errors import LoweringError
import unittest
from numba.extending import (
from numba.core.typing.templates import (
from .pdlike_usecase import Index, Series
def test_typing_vs_impl_signature_mismatch_handling_var_positional(self):
    """
        Tests that an overload which has a differing typing and implementing
        signature raises an exception and uses VAR_POSITIONAL (*args) in typing
        """

    def myoverload(a, kw=None):
        pass
    from .overload_usecases import var_positional_impl
    overload(myoverload)(var_positional_impl)

    @jit(nopython=True)
    def foo(a, b):
        return myoverload(a, b, 9, kw=11)
    with self.assertRaises(errors.TypingError) as e:
        foo(1, 5)
    msg = str(e.exception)
    self.assertIn('VAR_POSITIONAL (e.g. *args) argument kind', msg)
    self.assertIn("offending argument name is '*star_args_token'", msg)