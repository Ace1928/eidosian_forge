import builtins
import collections
import dis
import operator
import logging
import textwrap
from numba.core import errors, ir, config
from numba.core.errors import NotDefinedError, UnsupportedError, error_extras
from numba.core.ir_utils import get_definition, guard
from numba.core.utils import (PYVERSION, BINOPS_TO_OPERATORS,
from numba.core.byteflow import Flow, AdaptDFA, AdaptCFA, BlockKind
from numba.core.unsafe import eh
from numba.cpython.unsafe.tuple import unpack_single_tuple
def _insert_exception_variables(self):
    """Insert IR-nodes to initialize the exception variables.
        """
    tryblk = self.dfainfo.active_try_block
    endblk = tryblk['end']
    edgepushed = self.dfainfo.outgoing_edgepushed.get(endblk)
    if edgepushed:
        const_none = ir.Const(value=None, loc=self.loc)
        for var in edgepushed:
            if var in self.definitions:
                raise AssertionError('exception variable CANNOT be defined by other code')
            self.store(value=const_none, name=var)
            self._exception_vars.add(var)