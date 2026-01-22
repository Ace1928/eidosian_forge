import contextlib
import ctypes
import importlib
import inspect
import sys
import types
from typing import Any, Callable, Dict, List, Type, Union
import torch._C
import torch.utils._pytree as pytree
from torch import _utils_internal
from torch._functorch.pyfunctorch import dispatch_functorch
class _OpNamespace(types.ModuleType):
    """
    An op namespace to dynamically bind Operators into Python.

    Say a user has created a custom Operator called "my_namespace::my_op". To
    call this op, the user will write torch.ops.my_namespace.my_op(...).
    At startup, this operation will not yet be bound into Python. Instead, the
    following sequence of magic tricks will occur:
    1. `torch.ops.my_namespace` will invoke the `__getattr__` magic method
       on the `torch.ops` object, which will create a new `_OpNamespace`
       object called `my_namespace` and set it as an attribute on the `ops`
       object.
    2. `torch.ops.my_namespace.my_op` will then invoke `__getattr__` on
       the `my_namespace` object, which will retrieve the operation via
       `torch.get_operation`, a function bound from C++, and then in a similar
       fashion bind this new object onto the `my_namespace` object.
    3. `torch.ops.my_namespace.my_op(...)` then calls this new operation
        and subsequent accesses will incur no further lookup (the namespace and
        operation will already exist).
    """

    def __init__(self, name):
        super().__init__('torch.ops.' + name)
        self.name = name
        self._dir = []

    def __iter__(self):
        return iter(self._dir)

    def __getattr__(self, op_name):
        if op_name == '__file__':
            return 'torch.ops'
        elif op_name in ['__origin__', '__self__']:
            raise AttributeError(f"Invalid attribute '{op_name}' for '_OpNamespace' '{self.name}'")
        namespace_name = self.name
        qualified_op_name = f'{namespace_name}::{op_name}'
        try:
            op, overload_names = torch._C._jit_get_operation(qualified_op_name)
            if op is None:
                raise AttributeError(f"'_OpNamespace' '{self.name}' object has no attribute '{op_name}'")
        except RuntimeError as e:
            raise AttributeError(f"'_OpNamespace' '{self.name}' object has no attribute '{op_name}'") from e
        torch.jit._builtins._register_builtin(op, qualified_op_name)
        op.__module__ = self.__module__ + '.' + namespace_name
        opoverloadpacket = OpOverloadPacket(qualified_op_name, op_name, op, overload_names)
        opoverloadpacket.__module__ = self.__module__ + '.' + namespace_name
        setattr(self, op_name, opoverloadpacket)
        self._dir.append(op_name)
        return opoverloadpacket