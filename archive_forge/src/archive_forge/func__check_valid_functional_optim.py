from typing import Any, Callable, List, no_type_check
import torch
import torch.distributed as dist
from torch.autograd import Variable
from functools import partial
from dataclasses import dataclass
def _check_valid_functional_optim(self):
    if not hasattr(self.functional_optimizer, _FUNCTIONAL_OPTIM_STEP_METHOD_NAME):
        raise ValueError(f'Class {type(self.functional_optimizer)} must implement method {_FUNCTIONAL_OPTIM_STEP_METHOD_NAME}.')