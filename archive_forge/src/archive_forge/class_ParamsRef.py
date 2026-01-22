from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
class ParamsRef:
    """Set of parameters used to configure Solvers, Tactics and Simplifiers in Z3.

    Consider using the function `args2params` to create instances of this object.
    """

    def __init__(self, ctx=None, params=None):
        self.ctx = _get_ctx(ctx)
        if params is None:
            self.params = Z3_mk_params(self.ctx.ref())
        else:
            self.params = params
        Z3_params_inc_ref(self.ctx.ref(), self.params)

    def __deepcopy__(self, memo={}):
        return ParamsRef(self.ctx, self.params)

    def __del__(self):
        if self.ctx.ref() is not None and Z3_params_dec_ref is not None:
            Z3_params_dec_ref(self.ctx.ref(), self.params)

    def set(self, name, val):
        """Set parameter name with value val."""
        if z3_debug():
            _z3_assert(isinstance(name, str), 'parameter name must be a string')
        name_sym = to_symbol(name, self.ctx)
        if isinstance(val, bool):
            Z3_params_set_bool(self.ctx.ref(), self.params, name_sym, val)
        elif _is_int(val):
            Z3_params_set_uint(self.ctx.ref(), self.params, name_sym, val)
        elif isinstance(val, float):
            Z3_params_set_double(self.ctx.ref(), self.params, name_sym, val)
        elif isinstance(val, str):
            Z3_params_set_symbol(self.ctx.ref(), self.params, name_sym, to_symbol(val, self.ctx))
        elif z3_debug():
            _z3_assert(False, 'invalid parameter value')

    def __repr__(self):
        return Z3_params_to_string(self.ctx.ref(), self.params)

    def validate(self, ds):
        _z3_assert(isinstance(ds, ParamDescrsRef), 'parameter description set expected')
        Z3_params_validate(self.ctx.ref(), self.params, ds.descr)