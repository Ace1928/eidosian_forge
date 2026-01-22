import contextlib
from functools import partial
from typing import Any, Callable, Generator, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.plugins import Precision as FabricPrecision
from lightning_fabric.utilities.types import Steppable
from pytorch_lightning.core.hooks import CheckpointHooks
from pytorch_lightning.trainer import call
from pytorch_lightning.utilities import GradClipAlgorithmType
def _after_closure(self, model: 'pl.LightningModule', optimizer: Steppable) -> None:
    """Utility to share some code after the closure has been run."""
    trainer = model.trainer
    call._call_callback_hooks(trainer, 'on_before_optimizer_step', optimizer)
    call._call_lightning_module_hook(trainer, 'on_before_optimizer_step', optimizer)
    self._clip_gradients(model, optimizer, trainer.gradient_clip_val, gradient_clip_algorithm=trainer.gradient_clip_algorithm)