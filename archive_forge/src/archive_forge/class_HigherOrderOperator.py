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
class HigherOrderOperator(OperatorBase):

    def __init__(self, name):
        super().__init__()
        self._name = name
        self.__name__ = name
        _higher_order_ops[name] = self
        self._ns = 'higher_order'
        if self.__class__ is HigherOrderOperator:
            self_name_space = '.' + self.namespace if self.namespace else ''
            self.__module__ = self.__module__ + self_name_space
        self.non_fallthrough_keys = torch._C._dispatch_keyset_full()
        for dispatch_key in _HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS:
            self.fallthrough(dispatch_key)

    def py_impl(self, k):
        if isinstance(k, torch._C.DispatchKey) and (not self.non_fallthrough_keys.has(k)):
            self.non_fallthrough_keys = self.non_fallthrough_keys.add(k)
        return super().py_impl(k)

    @property
    def namespace(self):
        return self._ns

    def fallthrough(self, dispatch_key):
        self.non_fallthrough_keys = self.non_fallthrough_keys.remove(dispatch_key)

    def dispatch(self, dispatch_key, *args, **kwargs):
        from torch.utils._python_dispatch import _get_current_dispatch_mode
        if dispatch_key in self._dispatch_cache:
            kernel = self._dispatch_cache[dispatch_key]
            assert not isinstance(kernel, torch._C.DispatchKey)
            return kernel(*args, **kwargs)
        if dispatch_key == torch._C.DispatchKey.FuncTorchDynamicLayerFrontMode:
            return dispatch_functorch(self, args, kwargs)
        if dispatch_key == torch._C.DispatchKey.Python:
            from torch.utils._python_dispatch import _pop_mode_temporarily
            curr_mode = _get_current_dispatch_mode()
            assert curr_mode is not None, 'Illegal invocation of dispatch on torch._C.DispatchKey.Python without a mode.'
            assert type(curr_mode) in self.python_key_mode_table, f'Current active mode {curr_mode} not registered'
            handler = self.python_key_mode_table[type(curr_mode)]
            with _pop_mode_temporarily() as mode:
                return handler(mode, *args, **kwargs)
        functionality_key = torch._C._to_functionality_key(dispatch_key)
        if functionality_key in mode_stack_per_key():
            curr_stack = mode_stack_per_key()[functionality_key]
            if len(curr_stack) > 0 and (not torch._C._dispatch_tls_is_dispatch_key_excluded(DispatchKey.Python)):
                curr_mode = curr_stack[-1]
                pre_dispatch_modes = mode_stack_per_key().get(DispatchKey.PreDispatch, [])
                handler = self.python_key_mode_table[type(curr_mode)]
                if len(pre_dispatch_modes) > 0:
                    with temporarily_pop_mode(pre_dispatch_modes) as mode:
                        return handler(mode, *args, **kwargs)
        final_key = resolve_key(self, dispatch_key)
        if final_key not in self.py_kernels:
            raise NotImplementedError(f'could not find kernel for HigherOrderOperator {self._name} at dispatch key {final_key} (resolved from {dispatch_key})')
        self._dispatch_cache[dispatch_key] = self.py_kernels[final_key]
        kernel = self.py_kernels[final_key]
        assert not isinstance(kernel, torch._C.DispatchKey)
        return kernel(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        import torch._dynamo
        from torch._dynamo import disable

        @disable
        def wrapper():
            flat_args = _to_flat_tuple(args, kwargs)
            if torch.overrides.has_torch_function(flat_args):
                return torch.overrides.handle_torch_function(self, flat_args, *args, **kwargs)
            dispatch_key_set = _compute_keyset(args, kwargs, self.non_fallthrough_keys)
            return self.dispatch(dispatch_key_set.highestPriorityTypeId(), *args, **kwargs)
        return wrapper()

    def __str__(self):
        return f'{self.name()}'

    def name(self):
        return self._name