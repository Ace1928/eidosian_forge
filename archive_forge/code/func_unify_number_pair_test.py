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
def unify_number_pair_test(self, n):
    """
        Test all permutations of N-combinations of numeric types and ensure
        that the order of types in the sequence is irrelevant.
        """
    ctx = typing.Context()
    for tys in itertools.combinations(types.number_domain, n):
        res = [ctx.unify_types(*comb) for comb in itertools.permutations(tys)]
        first_result = res[0]
        self.assertIsInstance(first_result, types.Number)
        for other in res[1:]:
            self.assertEqual(first_result, other)