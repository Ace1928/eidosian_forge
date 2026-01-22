import collections
import functools
import inspect
import operator
import types
from typing import Dict, List, Optional
import torch
import torch.fx
from ..._guards import Source
from .. import polyfill, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..source import AttrSource, GetItemSource
from ..utils import (
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .functions import UserFunctionVariable, UserMethodVariable
class SizeVariable(TupleVariable):
    """torch.Size(...)"""

    def __init__(self, items: List[VariableTracker], proxy: Optional[torch.fx.Proxy]=None, **kwargs):
        self.proxy = proxy
        super().__init__(items, **kwargs)

    def python_type(self):
        return torch.Size

    def as_proxy(self):
        if self.proxy is not None:
            return self.proxy
        tracer = None
        proxies = self._as_proxy()
        for proxy in proxies:
            if isinstance(proxy, torch.fx.Proxy):
                tracer = proxy.tracer
                break
        if tracer is None:
            return torch.Size(proxies)
        proxy = tracer.create_proxy('call_function', torch.Size, (proxies,), {})
        proxy.node.meta['example_value'] = torch.Size([p.node.meta['example_value'] if not isinstance(p, int) else p for p in proxies])
        return proxy

    def reconstruct(self, codegen):
        codegen.load_import_from('torch', 'Size')
        codegen.foreach(self.items)
        build_torch_size = [create_instruction('BUILD_TUPLE', arg=len(self.items))] + create_call_function(1, True)
        return build_torch_size

    def unpack_var_sequence(self, tx):
        return list(self.items)

    def numel(self, tx):
        from .builtin import BuiltinVariable
        from .tensor import SymNodeVariable
        const_result = 1
        sym_sizes = []
        for v in self.items:
            if isinstance(v, ConstantVariable):
                const_result *= v.value
            else:
                assert isinstance(v, SymNodeVariable), type(v)
                sym_sizes.append(v)
        result = ConstantVariable.create(const_result)
        if sym_sizes and const_result == 1:
            result, *sym_sizes = sym_sizes
        if not sym_sizes or const_result == 0:
            return result
        mul = BuiltinVariable(operator.mul)
        for v in sym_sizes:
            result = mul.call_function(tx, [result, v], {})
        return result

    def call_method(self, tx, name, args: List['VariableTracker'], kwargs: Dict[str, 'VariableTracker']) -> 'VariableTracker':
        if name == '__getitem__':
            assert not kwargs and len(args) == 1
            out = self.get_item_dyn(tx, args[0])
            return out
        elif name == 'numel':
            assert not args and (not kwargs)
            return self.numel(tx)
        return super().call_method(tx, name, args, kwargs)

    def get_item_dyn(self, tx, arg: VariableTracker):
        from .tensor import SymNodeVariable
        if isinstance(arg, SymNodeVariable):
            index = arg.sym_num
        else:
            index = arg.as_python_constant()
        if isinstance(index, slice):
            return SizeVariable(self.items[index])
        else:
            assert isinstance(index, (int, torch.SymInt))
            return self.items[index]