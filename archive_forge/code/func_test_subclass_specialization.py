from itertools import product, permutations
from collections import defaultdict
import unittest
from numba.core.base import OverloadSelector
from numba.core.registry import cpu_target
from numba.core.imputils import builtin_registry, RegistryLoader
from numba.core import types
from numba.core.errors import NumbaNotImplementedError, NumbaTypeError
def test_subclass_specialization(self):
    os = OverloadSelector()
    self.assertTrue(issubclass(types.Sequence, types.Container))
    os.append(1, (types.Container, types.Container))
    lstty = types.List(types.boolean)
    self.assertEqual(os.find((lstty, lstty)), 1)
    os.append(2, (types.Container, types.Sequence))
    self.assertEqual(os.find((lstty, lstty)), 2)