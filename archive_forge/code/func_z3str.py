import functools
import logging
import math
import operator
import sympy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch._dynamo.exc import TorchDynamoException
from torch.fx.node import Argument, Target
from torch.utils._sympy.interp import sympy_interp
from torch.fx.experimental import _config as config
def z3str(e: z3.ExprRef) -> str:
    assert z3.is_expr(e), f'unsupported expression type: {e}'

    def get_args_str(e: z3.ExprRef) -> List[str]:
        return [z3str(e.arg(i)) for i in range(e.num_args())]
    e = z3.simplify(e)
    if not z3.is_app(e):
        raise ValueError(f"can't print Z3 expression: {e}")
    if z3.is_int_value(e) or z3.is_rational_value(e):
        return e.as_string()
    decl = e.decl()
    kind = decl.kind()
    op = str(decl)
    args = get_args_str(e)
    if kind == z3.Z3_OP_POWER:
        op = 'pow'
    elif kind in (z3.Z3_OP_ADD, z3.Z3_OP_MUL):

        def collect_str_args(e):
            if not (z3.is_app(e) and e.decl().kind() == kind):
                return [z3str(e)]
            else:
                return [x for i in range(e.num_args()) for x in collect_str_args(e.arg(i))]
        args = collect_str_args(e)
    elif kind == z3.Z3_OP_NOT:
        assert e.num_args() == 1
        arg = e.arg(0)
        assert z3.is_app(arg)
        argkind = arg.decl().kind()
        logic_inverse = {z3.Z3_OP_EQ: '!=', z3.Z3_OP_LE: '>', z3.Z3_OP_GE: '<'}
        if argkind in logic_inverse:
            op = logic_inverse[argkind]
            args = get_args_str(arg)
    elif kind in (z3.Z3_OP_TO_INT, z3.Z3_OP_TO_REAL):
        assert e.num_args() == 1
        argstr = z3str(e.arg(0))
        if argstr.startswith('(/'):
            return '(idiv' + argstr[2:]
        return argstr
    elif kind == z3.Z3_OP_UNINTERPRETED:
        assert e.num_args() == 0
        return str(decl)
    string = op + ' ' + ' '.join(args)
    return f'({string.rstrip()})'