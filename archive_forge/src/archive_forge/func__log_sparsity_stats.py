import inspect
import logging
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch.nn.utils.prune as pytorch_prune
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor, nn
from typing_extensions import TypedDict, override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_only
@rank_zero_only
def _log_sparsity_stats(self, prev: List[Tuple[int, int]], curr: List[Tuple[int, int]], amount: Union[int, float]=0) -> None:
    total_params = sum((p.numel() for layer, _ in self._parameters_to_prune for p in layer.parameters()))
    prev_total_zeros = sum((zeros for zeros, _ in prev))
    curr_total_zeros = sum((zeros for zeros, _ in curr))
    log.info(f'Applied `{self._pruning_method_name}`. Pruned: {prev_total_zeros}/{total_params} ({prev_total_zeros / total_params:.2%}) -> {curr_total_zeros}/{total_params} ({curr_total_zeros / total_params:.2%})')
    if self._verbose == 2:
        for i, (module, name) in enumerate(self._parameters_to_prune):
            prev_mask_zeros, prev_mask_size = prev[i]
            curr_mask_zeros, curr_mask_size = curr[i]
            log.info(f'Applied `{self._pruning_method_name}` to `{module!r}.{name}` with amount={amount}. Pruned: {prev_mask_zeros} ({prev_mask_zeros / prev_mask_size:.2%}) -> {curr_mask_zeros} ({curr_mask_zeros / curr_mask_size:.2%})')