import logging
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Union
import torch
from torch.nn import Module, ModuleDict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.optimizer import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
@staticmethod
def _apply_mapping_to_param_groups(param_groups: List[Dict[str, Any]], mapping: dict) -> List[Dict[str, Any]]:
    output = []
    for g in param_groups:
        group_state = {k: v for k, v in g.items() if k != 'params'}
        group_state['params'] = [mapping[p] for p in g['params']]
        output.append(group_state)
    return output