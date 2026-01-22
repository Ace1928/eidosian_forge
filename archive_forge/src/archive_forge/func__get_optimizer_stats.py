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
def _get_optimizer_stats(self, optimizer: Optimizer, names: List[str]) -> Dict[str, float]:
    stats = {}
    param_groups = optimizer.param_groups
    use_betas = 'betas' in optimizer.defaults
    for pg, name in zip(param_groups, names):
        lr = self._extract_lr(pg, name)
        stats.update(lr)
        momentum = self._extract_momentum(param_group=pg, name=name.replace(name, f'{name}-momentum'), use_betas=use_betas)
        stats.update(momentum)
        weight_decay = self._extract_weight_decay(pg, f'{name}-weight_decay')
        stats.update(weight_decay)
    return stats