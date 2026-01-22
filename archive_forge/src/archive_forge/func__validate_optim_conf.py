from contextlib import contextmanager
from dataclasses import fields
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, overload
from weakref import proxy
import torch
from torch import optim
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.types import Optimizable, ReduceLROnPlateau, _Stateful
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import LRSchedulerConfig, LRSchedulerTypeTuple
def _validate_optim_conf(optim_conf: Dict[str, Any]) -> None:
    valid_keys = {'optimizer', 'lr_scheduler', 'monitor'}
    extra_keys = optim_conf.keys() - valid_keys
    if extra_keys:
        rank_zero_warn(f'Found unsupported keys in the optimizer configuration: {set(extra_keys)}', category=RuntimeWarning)