import math
import warnings
from functools import partial
from typing import Callable, Iterable, Optional, Tuple, Union
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from .trainer_utils import SchedulerType
from .utils import logging
from .utils.versions import require_version
@staticmethod
def _get_options(param_group, param_shape):
    factored = len(param_shape) >= 2
    use_first_moment = param_group['beta1'] is not None
    return (factored, use_first_moment)