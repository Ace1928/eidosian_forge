import torch
from torch._jit_internal import _Await
from torch.jit._builtins import _register_builtin
from torch.utils import set_module
def _awaitable(func, *args, **kwargs):
    """Create Await object that will call specified functioni with specified args, when it is requested for the result."""
    return torch._C._awaitable(func, *args, **kwargs)