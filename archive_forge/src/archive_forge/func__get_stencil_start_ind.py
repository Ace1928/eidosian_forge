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
def _get_stencil_start_ind(self, start_length, gen_nodes, scope, loc):
    if isinstance(start_length, int):
        return abs(min(start_length, 0))

    def get_start_ind(s_length):
        return abs(min(s_length, 0))
    f_ir = compile_to_numba_ir(get_start_ind, {}, self.typingctx, self.targetctx, (types.intp,), self.typemap, self.calltypes)
    assert len(f_ir.blocks) == 1
    block = f_ir.blocks.popitem()[1]
    replace_arg_nodes(block, [start_length])
    gen_nodes += block.body[:-2]
    ret_var = block.body[-2].value.value
    return ret_var