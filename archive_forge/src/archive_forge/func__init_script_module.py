import torch
from ..modules import Module
from . import comm
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Sequence, Set, TypeVar, Union, cast
from torch._utils import _get_device_index
from collections import OrderedDict
def _init_script_module() -> 'torch.jit.ScriptModule':
    import torch.jit
    return torch.jit.ScriptModule()