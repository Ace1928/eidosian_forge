import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple
import torch
def _get_composite_method(cls, module, name, *args, **kwargs):
    old_method = None
    found = 0
    hooks_to_remove = []
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, BasePruningMethod) and hook._tensor_name == name:
            old_method = hook
            hooks_to_remove.append(k)
            found += 1
    assert found <= 1, f'Avoid adding multiple pruning hooks to the                same tensor {name} of module {module}. Use a PruningContainer.'
    for k in hooks_to_remove:
        del module._forward_pre_hooks[k]
    method = cls(*args, **kwargs)
    method._tensor_name = name
    if old_method is not None:
        if isinstance(old_method, PruningContainer):
            old_method.add_pruning_method(method)
            method = old_method
        elif isinstance(old_method, BasePruningMethod):
            container = PruningContainer(old_method)
            container.add_pruning_method(method)
            method = container
    return method