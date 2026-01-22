import abc
import os.path
from contextlib import contextmanager
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel.models import ComplexModel, UniTupleModel
from numba.core import config
def _di_subprogram(self, name, linkagename, line, function, argmap):
    return self.module.add_debug_info('DISubprogram', {'name': name, 'linkageName': linkagename, 'scope': self.difile, 'file': self.difile, 'line': line, 'type': self._di_subroutine_type(line, function, argmap), 'isLocal': False, 'isDefinition': True, 'scopeLine': line, 'isOptimized': config.OPT != 0, 'unit': self.dicompileunit}, is_distinct=True)