import logging
import traceback
import warnings
import weakref
from enum import auto, Enum
from functools import partial
from typing import (
import torch
import torch.distributed as dist
import torch.distributed.fsdp._flat_param as flat_param_file
import torch.nn as nn
from torch.distributed._composable_state import _get_module_state, _State
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
from torch.distributed.utils import _apply_to_tensors
from torch.utils._mode_utils import no_dispatch
from .api import (
@no_type_check
def _assert_in_training_states(state: _FSDPState, training_states: List[TrainingState]) -> None:
    """Asserts that FSDP is in the states ``_training_states``."""
    if state.training_state not in training_states:
        msg = f'expected to be in states {training_states} but current state is {state.training_state}'
        if state.rank == 0:
            if isinstance(state, nn.Module):
                print(f'Asserting FSDP instance is: {state}')
            print(f'ERROR: {msg}')
            traceback.print_stack()
        raise ValueError(msg)