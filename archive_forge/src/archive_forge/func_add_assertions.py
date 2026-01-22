import copy
import math
import operator
import traceback
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Set, Tuple
import sympy
import torch
import torch.fx
from torch.fx.experimental.symbolic_shapes import SymInt
from torch._export.pass_base import _ExportPassBase, ProxyValue, PassResult
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._sympy.value_ranges import ValueRanges
def add_assertions(val):
    call_backs: List[Callable] = []
    messages: List[str] = []
    if isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        symbol = val.node._expr
        if isinstance(symbol, sympy.Symbol) and symbol.name.startswith('i'):
            if symbol in self._asserts_generated_unbacked_symbols:
                return (call_backs, messages)
            constraint = self.range_constraints[symbol]
            min_val, max_val = _convert_range_to_int(constraint)
            assert_msg = f' is outside of inline constraint [{min_val}, {max_val}].'
            call_backs.append(partial(self._assert_range_constraint, lower=min_val, upper=max_val))
            messages.append(assert_msg)
            self._asserts_generated_unbacked_symbols.add(symbol)
    elif isinstance(val, torch.Tensor):
        for i, sym in enumerate(val.shape):
            cbs, msgs = add_assertions(sym)
            for cb, msg in zip(cbs, msgs):

                def sym_size_cb(proxy, assert_msg, dim):
                    dim_proxy = super(_AddRuntimeAssertionsForInlineConstraintsPass, self).call_operator(torch.ops.aten.sym_size.int, (proxy, dim), {}, self._create_dummy_node_metadata())
                    cb(proxy=dim_proxy, assert_msg=assert_msg)
                call_backs.append(partial(sym_size_cb, dim=i))
                messages.append(f'.shape[{i}]' + msg)
    return (call_backs, messages)