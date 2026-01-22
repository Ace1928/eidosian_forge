import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Tuple, TypeVar, Union
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
import pytorch_lightning as pl
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.strategies import _StrategyRegistry
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.distributed import ReduceOp
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.init import _EmptyInit
from lightning_fabric.utilities.optimizer import _optimizer_to_device, _optimizers_to_device
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.core.optimizer import LightningOptimizer, _init_optimizers_and_lr_schedulers
from pytorch_lightning.plugins import TorchCheckpointIO
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
from pytorch_lightning.plugins.precision import Precision
from pytorch_lightning.strategies.launchers.launcher import _Launcher
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.types import STEP_OUTPUT, LRSchedulerConfig
class _ForwardRedirection:
    """Implements the `forward-redirection`.

    A method call to a wrapped module gets rerouted through the wrapper's `forward` method instead.

    """

    def __call__(self, wrapper_module: Module, original_module: 'pl.LightningModule', method_name: str, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """Reroutes a method call through the `wrapper_module`'s `forward` method.

        Args:
            wrapper_module: The module that has `original_module` wrapped.
            original_module: The module that was wrapped inside `wrapper_module`.
            method_name: The name of the method that should be called on the `original_module` after inputs get
                redirected through the `wrapper_module`'s `forward` method.
            *args: The positional arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.
            **kwargs: The keyword arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.

        """
        assert method_name != 'forward'
        original_forward = original_module.forward

        def wrapped_forward(*_args: Any, **_kwargs: Any) -> Any:
            original_module.forward = original_forward
            method = getattr(original_module, method_name)
            out = method(*_args, **_kwargs)
            self.on_after_inner_forward(wrapper_module, original_module)
            return out
        original_module.forward = wrapped_forward
        wrapper_output = wrapper_module(*args, **kwargs)
        self.on_after_outer_forward(wrapper_module, original_module)
        return wrapper_output

    def on_after_inner_forward(self, wrapper_module: Module, original_module: 'pl.LightningModule') -> None:
        pass

    def on_after_outer_forward(self, wrapper_module: Module, original_module: 'pl.LightningModule') -> None:
        pass