import os, sys, subprocess
import dis
import itertools
import numpy as np
import numba
from numba import jit, njit
from numba.core import errors, ir, types, typing, typeinfer, utils
from numba.core.typeconv import Conversion
from numba.extending import overload_method
from numba.tests.support import TestCase, tag
from numba.tests.test_typeconv import CompatibilityTestMixin
from numba.core.untyped_passes import TranslateByteCode, IRProcessing
from numba.core.typed_passes import PartialTypeInference
from numba.core.compiler_machinery import FunctionPass, register_pass
import unittest
class TestPartialTypingErrors(unittest.TestCase):
    """
    Make sure partial typing stores type errors in compiler state properly
    """

    def test_partial_typing_error(self):

        def impl(flag):
            if flag:
                a = 1
            else:
                a = str(1)
            return a
        typing_errs = get_func_typing_errs(impl, (types.bool_,))
        self.assertTrue(isinstance(typing_errs, list) and len(typing_errs) == 1)
        self.assertTrue(isinstance(typing_errs[0], errors.TypingError) and 'Cannot unify' in typing_errs[0].msg)