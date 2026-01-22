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
def _configure_schedulers_manual_opt(schedulers: list) -> List[LRSchedulerConfig]:
    """Convert each scheduler into `LRSchedulerConfig` structure with relevant information, when using manual
    optimization."""
    lr_scheduler_configs = []
    for scheduler in schedulers:
        if isinstance(scheduler, dict):
            invalid_keys = {'reduce_on_plateau', 'monitor', 'strict'}
            keys_to_warn = [k for k in scheduler if k in invalid_keys]
            if keys_to_warn:
                rank_zero_warn(f'The lr scheduler dict contains the key(s) {keys_to_warn}, but the keys will be ignored. You need to call `lr_scheduler.step()` manually in manual optimization.', category=RuntimeWarning)
            config = LRSchedulerConfig(**{key: scheduler[key] for key in scheduler if key not in invalid_keys})
        else:
            config = LRSchedulerConfig(scheduler)
        lr_scheduler_configs.append(config)
    return lr_scheduler_configs