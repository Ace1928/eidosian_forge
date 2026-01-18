from llvmlite import ir, binding as ll
from numba.core import types, datamodel
from numba.core.datamodel.testing import test_factory
from numba.core.datamodel.manager import DataModelManager
from numba.core.datamodel.models import OpaqueModel
import unittest
def test_two_arrays(self):
    fe_args = [types.Array(types.int32, 1, 'C')] * 2
    self._test_as_arguments(fe_args)