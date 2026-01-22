import torch
from torch._C import _rename_privateuse1_backend, _get_privateuse1_backend_name
from typing import List, Optional, Union
def _get_custom_mod_func(func_name: str):
    """
    Return the func named `func_name` defined in custom device module. If not defined,
    return `None`. And the func is registered with `torch.utils.rename_privateuse1_backend('foo')`
    and `torch._register_device_module('foo', BackendModule)`.
    If the custom device module or the func is not defined, it will give warning or error message.
    Args:
        func_name (str): return the callable func named func_name defined in custom device module.
    Example::
        class DummyfooModule:
            @staticmethod
            def is_available():
                return True
            @staticmethod
            def func_name(*args, **kwargs):
                ....
        torch.utils.rename_privateuse1_backend("foo")
        torch._register_device_module("foo", DummyfooModule)
        foo_is_available_func = torch.utils.backend_registration._get_custom_mod_func("is_available")
        if foo_is_available_func:
            foo_is_available = foo_is_available_func()
        func_ = torch.utils.backend_registration._get_custom_mod_func("func_name")
        if func_:
            result = func_(*args, **kwargs)
    Attention: This function is not meant to be used directly by users, which is why
    it is marked as private. It is a convenience function for backend implementers to
    more easily call the hooks into their backend extensions.
    """
    assert isinstance(func_name, str), f'func_name must be `str`, but got `{type(func_name)}`.'
    backend_name = _get_privateuse1_backend_name()
    custom_device_mod = getattr(torch, backend_name, None)
    function = getattr(custom_device_mod, func_name, None)
    if custom_device_mod is None or function is None:
        message = f'Try to call torch.{backend_name}.{func_name}. The backend must register a custom backend '
        message += f"module with `torch._register_device_module('{backend_name}', BackendModule)`. And "
        message += f"BackendModule needs to have the following API's:\n `{func_name}(*args, **kwargs)`. \n"
        raise RuntimeError(message)
    return function