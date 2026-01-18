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
def unfreeze_and_add_param_group(modules: Union[Module, Iterable[Union[Module, Iterable]]], optimizer: Optimizer, lr: Optional[float]=None, initial_denom_lr: float=10.0, train_bn: bool=True) -> None:
    """Unfreezes a module and adds its parameters to an optimizer.

        Args:
            modules: A module or iterable of modules to unfreeze.
                Their parameters will be added to an optimizer as a new param group.
            optimizer: The provided optimizer will receive new parameters and will add them to
                `add_param_group`
            lr: Learning rate for the new param group.
            initial_denom_lr: If no lr is provided, the learning from the first param group will be used
                and divided by `initial_denom_lr`.
            train_bn: Whether to train the BatchNormalization layers.

        """
    BaseFinetuning.make_trainable(modules)
    params_lr = optimizer.param_groups[0]['lr'] if lr is None else float(lr)
    denom_lr = initial_denom_lr if lr is None else 1.0
    params = BaseFinetuning.filter_params(modules, train_bn=train_bn, requires_grad=True)
    params = BaseFinetuning.filter_on_optimizer(optimizer, params)
    if params:
        optimizer.add_param_group({'params': params, 'lr': params_lr / denom_lr})