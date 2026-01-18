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
def z3op(op: Callable, validator: 'TranslationValidator') -> Callable:
    from torch.fx.experimental.sym_node import sym_sqrt
    boolean_ops = {operator.not_, operator.and_, operator.or_}
    as_bool = op in boolean_ops

    def lift(func):

        def wrap(a) -> z3.ExprRef:
            if isinstance(a, (z3.ArithRef, z3.BoolRef)):
                return a
            if isinstance(a, bool) or (as_bool and isinstance(a, int)):
                return z3.BoolVal(bool(a))
            if isinstance(a, (int, sympy.Integer)):
                return z3.IntVal(int(a))
            if isinstance(a, (float, sympy.Float)):
                return z3.RealVal(float(a))
            raise ValueError(f"can't lift type: {type(a)}")

        @functools.wraps(func)
        def wrapper(*args):
            wrapped_args = (wrap(a) for a in args)
            return func(*wrapped_args)
        return wrapper
    ops = _Z3Ops(validator)
    replacement_map = {operator.not_: lift(z3.Not), operator.and_: lift(z3.And), operator.or_: lift(z3.Or), operator.floordiv: lift(ops.floordiv), operator.truediv: lift(ops.div), operator.mod: lift(ops.mod), operator.abs: lift(ops.abs), math.ceil: lift(ops.ceil), math.floor: lift(ops.floor), torch.sym_float: lift(ops.to_real), torch.sym_max: lift(ops.max), torch.sym_min: lift(ops.min), torch.sym_ite: lift(lambda b, t, f: t if b else f), sym_sqrt: lift(ops.sqrt), torch._assert: torch._assert}
    return replacement_map[op] if op in replacement_map else lift(op)