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
def _extract_momentum(self, param_group: Dict[str, List], name: str, use_betas: bool) -> Dict[str, float]:
    if not self.log_momentum:
        return {}
    momentum = param_group['betas'][0] if use_betas else param_group.get('momentum', 0)
    self.last_momentum_values[name] = momentum
    return {name: momentum}