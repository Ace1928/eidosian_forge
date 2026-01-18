from itertools import product, permutations
from collections import defaultdict
import unittest
from numba.core.base import OverloadSelector
from numba.core.registry import cpu_target
from numba.core.imputils import builtin_registry, RegistryLoader
from numba.core import types
from numba.core.errors import NumbaNotImplementedError, NumbaTypeError
def test_select_and_sort_2(self):
    os = OverloadSelector()
    os.append(1, (types.Container,))
    os.append(2, (types.Sequence,))
    os.append(3, (types.MutableSequence,))
    os.append(4, (types.List,))
    compats = os._select_compatible((types.List,))
    self.assertEqual(len(compats), 4)
    ordered, scoring = os._sort_signatures(compats)
    self.assertEqual(len(ordered), 4)
    self.assertEqual(len(scoring), 4)
    self.assertEqual(ordered[0], (types.List,))
    self.assertEqual(scoring[types.List,], 0)
    self.assertEqual(scoring[types.MutableSequence,], 1)
    self.assertEqual(scoring[types.Sequence,], 2)
    self.assertEqual(scoring[types.Container,], 3)