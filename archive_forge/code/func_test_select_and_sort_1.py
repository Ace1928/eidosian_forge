from itertools import product, permutations
from collections import defaultdict
import unittest
from numba.core.base import OverloadSelector
from numba.core.registry import cpu_target
from numba.core.imputils import builtin_registry, RegistryLoader
from numba.core import types
from numba.core.errors import NumbaNotImplementedError, NumbaTypeError
def test_select_and_sort_1(self):
    os = OverloadSelector()
    os.append(1, (types.Any, types.Boolean))
    os.append(2, (types.Boolean, types.Integer))
    os.append(3, (types.Boolean, types.Any))
    os.append(4, (types.Boolean, types.Boolean))
    compats = os._select_compatible((types.boolean, types.boolean))
    self.assertEqual(len(compats), 3)
    ordered, scoring = os._sort_signatures(compats)
    self.assertEqual(len(ordered), 3)
    self.assertEqual(len(scoring), 3)
    self.assertEqual(ordered[0], (types.Boolean, types.Boolean))
    self.assertEqual(scoring[types.Boolean, types.Boolean], 0)
    self.assertEqual(scoring[types.Boolean, types.Any], 1)
    self.assertEqual(scoring[types.Any, types.Boolean], 1)