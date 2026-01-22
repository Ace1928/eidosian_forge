import collections
import logging
import os
import random
import types
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from packaging.version import Version
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import (
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.train._internal import session
from ray.train._internal.accelerator import Accelerator
from ray.train._internal.session import get_accelerator, set_accelerator
from ray.util.annotations import Deprecated, PublicAPI
@PublicAPI(stability='beta')
def accelerate(amp: bool=False) -> None:
    """Enables training optimizations.

    Arguments:
        amp: If true, perform training with automatic mixed precision.
            Otherwise, use full precision.

    .. warning:: ``train.torch.accelerate`` cannot be called more than once, and it
       must be called before any other ``train.torch`` utility function.
    """
    try:
        set_accelerator(_TorchAccelerator(amp=amp))
    except RuntimeError:
        raise RuntimeError('An accelerator has already been set. Make sure `train.torch.accelerate()` is not called multiple times, and is called before any of the prepare methods.')