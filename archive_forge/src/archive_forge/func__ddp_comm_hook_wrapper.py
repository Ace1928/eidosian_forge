from enum import Enum
from functools import partial
import torch.distributed as dist
from . import (
def _ddp_comm_hook_wrapper(comm_hook, model, state):
    model.register_comm_hook(state, comm_hook)