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
def _legalize_exception_vars(self):
    """Search for unsupported use of exception variables.
        Note, they cannot be stored into user variable.
        """
    excvars = self._exception_vars.copy()
    for varname, defnvars in self.definitions.items():
        for v in defnvars:
            if isinstance(v, ir.Var):
                k = v.name
                if k in excvars:
                    excvars.add(varname)
    uservar = list(filter(lambda x: not x.startswith('$'), excvars))
    if uservar:
        first = uservar[0]
        loc = self.current_scope.get(first).loc
        msg = 'Exception object cannot be stored into variable ({}).'
        raise errors.UnsupportedError(msg.format(first), loc=loc)