import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def _get_ptr_by_name(self, attrname):
    return self._get_ptr_by_index(self._namemap[attrname])