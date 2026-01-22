import operator
import warnings
from itertools import product
import numpy as np
from numba import njit, typeof, literally, prange
from numba.core import types, ir, ir_utils, cgutils, errors, utils
from numba.core.extending import (
from numba.core.cpu import InlineOptions
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.typed_passes import InlineOverloads
from numba.core.typing import signature
from numba.tests.support import (TestCase, unittest,
class TestInlineOptions(TestCase):

    def test_basic(self):
        always = InlineOptions('always')
        self.assertTrue(always.is_always_inline)
        self.assertFalse(always.is_never_inline)
        self.assertFalse(always.has_cost_model)
        self.assertEqual(always.value, 'always')
        never = InlineOptions('never')
        self.assertFalse(never.is_always_inline)
        self.assertTrue(never.is_never_inline)
        self.assertFalse(never.has_cost_model)
        self.assertEqual(never.value, 'never')

        def cost_model(x):
            return x
        model = InlineOptions(cost_model)
        self.assertFalse(model.is_always_inline)
        self.assertFalse(model.is_never_inline)
        self.assertTrue(model.has_cost_model)
        self.assertIs(model.value, cost_model)