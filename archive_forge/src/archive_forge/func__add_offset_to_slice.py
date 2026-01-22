import numbers
import copy
import types as pytypes
from operator import add
import operator
import numpy as np
import numba.parfors.parfor
from numba.core import types, ir, rewrites, config, ir_utils
from numba.core.typing.templates import infer_global, AbstractTemplate
from numba.core.typing import signature
from numba.core import  utils, typing
from numba.core.ir_utils import (get_call_table, mk_unique_var,
from numba.core.errors import NumbaValueError
from numba.core.utils import OPERATORS_TO_BUILTINS
from numba.np import numpy_support
def _add_offset_to_slice(self, slice_var, offset_var, out_nodes, scope, loc):
    if isinstance(slice_var, slice):
        f_text = 'def f(offset):\n                return slice({} + offset, {} + offset)\n            '.format(slice_var.start, slice_var.stop)
        loc = {}
        exec(f_text, {}, loc)
        f = loc['f']
        args = [offset_var]
        arg_typs = (types.intp,)
    else:

        def f(old_slice, offset):
            return slice(old_slice.start + offset, old_slice.stop + offset)
        args = [slice_var, offset_var]
        slice_type = self.typemap[slice_var.name]
        arg_typs = (slice_type, types.intp)
    _globals = self.func_ir.func_id.func.__globals__
    f_ir = compile_to_numba_ir(f, _globals, self.typingctx, self.targetctx, arg_typs, self.typemap, self.calltypes)
    _, block = f_ir.blocks.popitem()
    replace_arg_nodes(block, args)
    new_index = block.body[-2].value.value
    out_nodes.extend(block.body[:-2])
    return new_index