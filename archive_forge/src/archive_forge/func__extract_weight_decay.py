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
def _extract_weight_decay(self, param_group: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Extracts the weight decay statistics from a parameter group."""
    if not self.log_weight_decay:
        return {}
    weight_decay = param_group['weight_decay']
    self.last_weight_decay_values[name] = weight_decay
    return {name: weight_decay}