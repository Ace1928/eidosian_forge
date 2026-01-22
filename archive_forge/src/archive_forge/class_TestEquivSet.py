import itertools
import numpy as np
import sys
from collections import namedtuple
from io import StringIO
from numba import njit, typeof, prange
from numba.core import (
from numba.tests.support import (TestCase, tag, skip_parfors_unsupported,
from numba.parfors.array_analysis import EquivSet, ArrayAnalysis
from numba.core.compiler import Compiler, Flags, PassManager
from numba.core.ir_utils import remove_dead
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass
from numba.experimental import jitclass
import unittest
class TestEquivSet(TestCase):
    """
    Test array_analysis.EquivSet.
    """

    def test_insert_equiv(self):
        s1 = EquivSet()
        s1.insert_equiv('a', 'b')
        self.assertTrue(s1.is_equiv('a', 'b'))
        self.assertTrue(s1.is_equiv('b', 'a'))
        s1.insert_equiv('c', 'd')
        self.assertTrue(s1.is_equiv('c', 'd'))
        self.assertFalse(s1.is_equiv('c', 'a'))
        s1.insert_equiv('a', 'c')
        self.assertTrue(s1.is_equiv('a', 'b', 'c', 'd'))
        self.assertFalse(s1.is_equiv('a', 'e'))

    def test_intersect(self):
        s1 = EquivSet()
        s2 = EquivSet()
        r = s1.intersect(s2)
        self.assertTrue(r.is_empty())
        s1.insert_equiv('a', 'b')
        r = s1.intersect(s2)
        self.assertTrue(r.is_empty())
        s2.insert_equiv('b', 'c')
        r = s1.intersect(s2)
        self.assertTrue(r.is_empty())
        s2.insert_equiv('d', 'a')
        r = s1.intersect(s2)
        self.assertTrue(r.is_empty())
        s1.insert_equiv('a', 'e')
        s2.insert_equiv('c', 'd')
        r = s1.intersect(s2)
        self.assertTrue(r.is_equiv('a', 'b'))
        self.assertFalse(r.is_equiv('a', 'e'))
        self.assertFalse(r.is_equiv('c', 'd'))