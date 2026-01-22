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
def _make_pruning_permanent_on_state_dict(self, pl_module: LightningModule) -> Dict[str, Any]:
    state_dict = pl_module.state_dict()
    map_pruned_params = {k.replace('_mask', '') for k in state_dict if k.endswith('_mask')}
    for tensor_name in map_pruned_params:
        orig = state_dict.pop(tensor_name + '_orig')
        mask = state_dict.pop(tensor_name + '_mask')
        state_dict[tensor_name] = mask.to(dtype=orig.dtype) * orig

    def move_to_cpu(tensor: Tensor) -> Tensor:
        return tensor.cpu()
    return apply_to_collection(state_dict, Tensor, move_to_cpu)