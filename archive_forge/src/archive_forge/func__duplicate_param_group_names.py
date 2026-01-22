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
def _duplicate_param_group_names(self, param_groups: List[Dict]) -> Set[str]:
    names = [pg.get('name', f'pg{i}') for i, pg in enumerate(param_groups, start=1)]
    unique = set(names)
    if len(names) == len(unique):
        return set()
    return {n for n in names if names.count(n) > 1}