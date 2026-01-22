import abc
import os.path
from contextlib import contextmanager
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel.models import ComplexModel, UniTupleModel
from numba.core import config
def _di_location(self, line):
    return self.module.add_debug_info('DILocation', {'line': line, 'column': 1, 'scope': self.subprograms[-1]})