import functools
from typing import Dict, List, Mapping, Optional, Union
import torch
import torch.nn as nn
from .state import PartialState
from .utils import (
from .utils.modeling import get_non_persistent_buffers
from .utils.other import recursive_getattr
def add_hook_to_module(module: nn.Module, hook: ModelHook, append: bool=False):
    """
    Adds a hook to a given module. This will rewrite the `forward` method of the module to include the hook, to remove
    this behavior and restore the original `forward` method, use `remove_hook_from_module`.

    <Tip warning={true}>

    If the module already contains a hook, this will replace it with the new hook passed by default. To chain two hooks
    together, pass `append=True`, so it chains the current and new hook into an instance of the `SequentialHook` class.

    </Tip>

    Args:
        module (`torch.nn.Module`):
            The module to attach a hook to.
        hook (`ModelHook`):
            The hook to attach.
        append (`bool`, *optional*, defaults to `False`):
            Whether the hook should be chained with an existing one (if module already contains a hook) or not.

    Returns:
        `torch.nn.Module`: The same module, with the hook attached (the module is modified in place, so the result can
        be discarded).
    """
    if append and getattr(module, '_hf_hook', None) is not None:
        old_hook = module._hf_hook
        remove_hook_from_module(module)
        hook = SequentialHook(old_hook, hook)
    if hasattr(module, '_hf_hook') and hasattr(module, '_old_forward'):
        old_forward = module._old_forward
    else:
        old_forward = module.forward
        module._old_forward = old_forward
    module = hook.init_hook(module)
    module._hf_hook = hook

    def new_forward(module, *args, **kwargs):
        args, kwargs = module._hf_hook.pre_forward(module, *args, **kwargs)
        if module._hf_hook.no_grad:
            with torch.no_grad():
                output = module._old_forward(*args, **kwargs)
        else:
            output = module._old_forward(*args, **kwargs)
        return module._hf_hook.post_forward(module, output)
    if 'GraphModuleImpl' in str(type(module)):
        module.__class__.forward = functools.update_wrapper(functools.partial(new_forward, module), old_forward)
    else:
        module.forward = functools.update_wrapper(functools.partial(new_forward, module), old_forward)
    return module