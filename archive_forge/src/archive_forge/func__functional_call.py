import contextlib
import warnings
from collections import defaultdict
from typing import Any, Dict, Iterator, Optional, Set, Tuple, Union
import torch
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
def _functional_call(module: 'torch.nn.Module', parameters_and_buffers: Dict[str, Tensor], args: Union[Any, Tuple], kwargs: Optional[Dict[str, Any]]=None, *, tie_weights: bool=True, strict: bool=False):
    if torch.jit.is_tracing() or torch.jit.is_scripting() or isinstance(module, (torch.jit.RecursiveScriptModule, torch.jit.ScriptModule, torch.jit.ScriptFunction)):
        raise RuntimeError("The stateless API can't be used with Jitted modules")
    if isinstance(module, torch.nn.DataParallel):
        raise RuntimeError("The stateless API can't be used with nn.DataParallel module")
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, tuple):
        args = (args,)
    with _reparametrize_module(module, parameters_and_buffers, tie_weights=tie_weights, strict=strict):
        return module(*args, **kwargs)