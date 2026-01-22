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
def _inject_call(self, func, gv_name, res_name=None):
    """A helper function to inject a call to *func* which is a python
        function.
        Parameters
        ----------
        func : callable
            The function object to be called.
        gv_name : str
            The variable name to be used to store the function object.
        res_name : str; optional
            The variable name to be used to store the call result.
            If ``None``, a name is created automatically.
        """
    gv_fn = ir.Global(gv_name, func, loc=self.loc)
    self.store(value=gv_fn, name=gv_name, redefine=True)
    callres = ir.Expr.call(self.get(gv_name), (), (), loc=self.loc)
    res_name = res_name or '$callres_{}'.format(gv_name)
    self.store(value=callres, name=res_name, redefine=True)