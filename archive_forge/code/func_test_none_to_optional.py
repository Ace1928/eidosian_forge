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
def test_none_to_optional(self):
    """
        Test unification of `none` and multiple number types to optional type
        """
    ctx = typing.Context()
    for tys in itertools.combinations(types.number_domain, 2):
        tys = list(tys)
        expected = types.Optional(ctx.unify_types(*tys))
        results = [ctx.unify_types(*comb) for comb in itertools.permutations(tys + [types.none])]
        for res in results:
            self.assertEqual(res, expected)