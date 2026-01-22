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
@no_type_check
def _set_use_dtensor(fsdp_state: _FSDPState) -> None:
    if getattr(fsdp_state, '_device_mesh', None):
        state_dict_type = fsdp_state._state_dict_type
        if state_dict_type == StateDictType.LOCAL_STATE_DICT:
            raise RuntimeError('Found state_dict_type LOCAL_STATE_DICT', 'DeviceMesh is not compatible with LOCAL_STATE_DICT.', 'Please set state_dict_type to SHARDED_STATE_DICT to get DTensor state_dict.')
        elif state_dict_type == StateDictType.FULL_STATE_DICT:
            logger.warning('Found both state_dict_type FULL_STATE_DICT and device_mesh. Please set state_dict_type to SHARDED_STATE_DICT to get DTensor state_dict.')
        else:
            fsdp_state._state_dict_config._use_dtensor = True