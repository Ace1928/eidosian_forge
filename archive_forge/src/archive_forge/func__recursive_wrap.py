import contextlib
import copy
from abc import ABC, abstractmethod
from typing import (
import torch.nn as nn
def _recursive_wrap(module: nn.Module, auto_wrap_policy: Callable, wrapper_cls: Callable, ignored_modules: Set[nn.Module], ignored_params: Set[nn.Parameter], only_wrap_children: bool=False, **kwargs: Any) -> Tuple[nn.Module, int]:
    """
    Wraps submodules of ``module`` for which ``auto_wrap_policy`` returns
    ``True`` with ``wrapper_cls``.

    Args:
        module (nn.Module): Module to recursively wrap.
        auto_wrap_policy (Callable): A callable representing a policy that
            determines which modules to recursively wrap with ``wrapper_cls``.
        ignored_modules (Set[torch.nn.Module]): Modules to ignore when
            wrapping.
        ignored_params (Set[torch.nn.Parameter]): Parameters to ignore when
            wrapping; these should be the parameters contained in the modules
            in ``ignored_modules``.
    Returns:
        (nn.Module, int):
            ``module`` after wrapping and the numel recursively wrapped.
    """
    assert auto_wrap_policy is not None, 'Must specify auto_wrap_policy.'
    assert wrapper_cls is not None, 'Must specify wrapper_cls'
    for _, child in module.named_modules():
        if child in ignored_modules:
            continue
        try:
            assert not isinstance(child, cast(type, wrapper_cls))
        except TypeError:
            pass
    nonwrapped_numel = sum((p.numel() for p in module.parameters() if p not in ignored_params))
    assert auto_wrap_policy is not None
    if auto_wrap_policy(module=module, recurse=True, nonwrapped_numel=nonwrapped_numel):
        total_wrapped_numel = 0
        for name, child in module.named_children():
            if child in ignored_modules:
                continue
            wrapped_child, num_wrapped_params = _recursive_wrap(module=child, auto_wrap_policy=auto_wrap_policy, wrapper_cls=wrapper_cls, ignored_modules=ignored_modules, ignored_params=ignored_params, **kwargs)
            setattr(module, name, wrapped_child)
            total_wrapped_numel += num_wrapped_params
        remainder = nonwrapped_numel - total_wrapped_numel
        if not only_wrap_children and auto_wrap_policy(module=module, recurse=False, nonwrapped_numel=remainder):
            return (_wrap(module, wrapper_cls, **kwargs), nonwrapped_numel)
        else:
            return (module, total_wrapped_numel)
    return (module, 0)