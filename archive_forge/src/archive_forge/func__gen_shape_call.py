import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _gen_shape_call(self, equiv_set, var, ndims, shape, post):
    if isinstance(shape, ir.Var):
        shape = equiv_set.get_shape(shape)
    if isinstance(shape, ir.Var):
        attr_var = shape
        shape_attr_call = None
        shape = None
    elif isinstance(shape, ir.Arg):
        attr_var = var
        shape_attr_call = None
        shape = None
    else:
        shape_attr_call = ir.Expr.getattr(var, 'shape', var.loc)
        attr_var = ir.Var(var.scope, mk_unique_var('{}_shape'.format(var.name)), var.loc)
        shape_attr_typ = types.containers.UniTuple(types.intp, ndims)
    size_vars = []
    use_attr_var = False
    if shape:
        nshapes = len(shape)
        if ndims < nshapes:
            shape = shape[nshapes - ndims:]
    for i in range(ndims):
        skip = False
        if shape and shape[i]:
            if isinstance(shape[i], ir.Var):
                typ = self.typemap[shape[i].name]
                if isinstance(typ, (types.Number, types.SliceType)):
                    size_var = shape[i]
                    skip = True
            else:
                if isinstance(shape[i], int):
                    size_val = ir.Const(shape[i], var.loc)
                else:
                    size_val = shape[i]
                assert isinstance(size_val, ir.Const)
                size_var = ir.Var(var.scope, mk_unique_var('{}_size{}'.format(var.name, i)), var.loc)
                post.append(ir.Assign(size_val, size_var, var.loc))
                self._define(equiv_set, size_var, types.intp, size_val)
                skip = True
        if not skip:
            size_var = ir.Var(var.scope, mk_unique_var('{}_size{}'.format(var.name, i)), var.loc)
            getitem = ir.Expr.static_getitem(attr_var, i, None, var.loc)
            use_attr_var = True
            self.calltypes[getitem] = None
            post.append(ir.Assign(getitem, size_var, var.loc))
            self._define(equiv_set, size_var, types.intp, getitem)
        size_vars.append(size_var)
    if use_attr_var and shape_attr_call:
        post.insert(0, ir.Assign(shape_attr_call, attr_var, var.loc))
        self._define(equiv_set, attr_var, shape_attr_typ, shape_attr_call)
    return tuple(size_vars)