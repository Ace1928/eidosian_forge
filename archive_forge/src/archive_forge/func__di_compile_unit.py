import abc
import os.path
from contextlib import contextmanager
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel.models import ComplexModel, UniTupleModel
from numba.core import config
def _di_compile_unit(self):
    return self.module.add_debug_info('DICompileUnit', {'language': ir.DIToken('DW_LANG_C_plus_plus'), 'file': self.difile, 'producer': 'clang (Numba)', 'runtimeVersion': 0, 'isOptimized': config.OPT != 0, 'emissionKind': ir.DIToken(self.emission_kind)}, is_distinct=True)