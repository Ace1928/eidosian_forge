from llvmlite import ir, binding as ll
from numba.core import types, datamodel
from numba.core.datamodel.testing import test_factory
from numba.core.datamodel.manager import DataModelManager
from numba.core.datamodel.models import OpaqueModel
import unittest
def test_tuple_of_array(self):
    ty = types.UniTuple(dtype=types.int32[:], count=2)
    dm = self.dmm[ty]
    self.assertTrue(dm.contains_nrt_meminfo())