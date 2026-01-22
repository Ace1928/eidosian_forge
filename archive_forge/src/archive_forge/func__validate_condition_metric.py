import logging
from typing import Any, Callable, Dict, Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_prefixed_message, rank_zero_warn
def _validate_condition_metric(self, logs: Dict[str, Tensor]) -> bool:
    monitor_val = logs.get(self.monitor)
    error_msg = f'Early stopping conditioned on metric `{self.monitor}` which is not available. Pass in or modify your `EarlyStopping` callback to use any of the following: `{'`, `'.join(list(logs.keys()))}`'
    if monitor_val is None:
        if self.strict:
            raise RuntimeError(error_msg)
        if self.verbose > 0:
            rank_zero_warn(error_msg, category=RuntimeWarning)
        return False
    return True