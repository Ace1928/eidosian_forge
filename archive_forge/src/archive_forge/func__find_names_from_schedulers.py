import itertools
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Set, Tuple, Type
import torch
from torch.optim.optimizer import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.types import LRSchedulerConfig
def _find_names_from_schedulers(self, lr_scheduler_configs: List[LRSchedulerConfig]) -> Tuple[List[List[str]], List[Optimizer], DefaultDict[Type[Optimizer], int]]:
    names = []
    seen_optimizers: List[Optimizer] = []
    seen_optimizer_types: DefaultDict[Type[Optimizer], int] = defaultdict(int)
    for config in lr_scheduler_configs:
        sch = config.scheduler
        name = config.name if config.name is not None else 'lr-' + sch.optimizer.__class__.__name__
        updated_names = self._check_duplicates_and_update_name(sch.optimizer, name, seen_optimizers, seen_optimizer_types, config)
        names.append(updated_names)
    return (names, seen_optimizers, seen_optimizer_types)