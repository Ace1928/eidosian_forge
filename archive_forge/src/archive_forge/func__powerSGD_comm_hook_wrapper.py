from enum import Enum
from functools import partial
import torch.distributed as dist
from . import (
def _powerSGD_comm_hook_wrapper(comm_hook, model, state, matrix_approximation_rank, start_powerSGD_iter=1000):
    """
    To be consistent with the wrappers of other DDP comm hooks, the input state only needs to be a process group,
    which will be wrapped up with other state info.
    """
    powerSGD_state = powerSGD.PowerSGDState(process_group=state, matrix_approximation_rank=matrix_approximation_rank, start_powerSGD_iter=start_powerSGD_iter)
    model.register_comm_hook(powerSGD_state, comm_hook)