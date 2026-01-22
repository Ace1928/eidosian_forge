import contextlib
import logging
import math
import warnings
from typing import Any, Callable, cast, Dict, Generator, Iterator, no_type_check, Tuple
import torch
import torch.distributed as dist
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as checkpoint_wrapper
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._shard.sharded_tensor import (
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import _mesh_resources
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._debug_utils import SimpleProfiler
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.fsdp.api import (
from torch.distributed.utils import _replace_by_prefix
from ._fsdp_extensions import (
from ._unshard_param_utils import _unshard_fsdp_state_params, FLAT_PARAM
def _register_all_state_dict_hooks(state: _FSDPState):
    """
    Registers pre-save, post-save, pre-load, and post-load state dict hooks.
    """
    for hook_registration_fn_str, hook, hook_registration_fn_kwargs in (('register_state_dict_pre_hook', _pre_state_dict_hook, {}), ('_register_state_dict_hook', _post_state_dict_hook, {}), ('_register_load_state_dict_pre_hook', _pre_load_state_dict_hook, {'with_module': True}), ('register_load_state_dict_post_hook', _post_load_state_dict_hook, {})):
        _register_state_dict_hooks_base(state, hook_registration_fn_str, hook, hook_registration_fn_kwargs)