import importlib
import logging
import os
import uuid
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast
import torch
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.types import _TORCH_LRSCHEDULER
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.parsing import lightning_hasattr, lightning_setattr
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.types import STEP_OUTPUT, LRScheduler, LRSchedulerConfig
class _LRCallback(Callback):
    """Special callback used by the learning rate finder. This callback logs the learning rate before each batch and
    logs the corresponding loss after each batch.

    Args:
        num_training: number of iterations done by the learning rate finder
        early_stop_threshold: threshold for stopping the search. If the
            loss at any point is larger than ``early_stop_threshold*best_loss``
            then the search is stopped. To disable, set to ``None``.
        progress_bar_refresh_rate: rate to refresh the progress bar for
            the learning rate finder
        beta: smoothing value, the loss being logged is a running average of
            loss values logged until now. ``beta`` controls the forget rate i.e.
            if ``beta=0`` all past information is ignored.

    """

    def __init__(self, num_training: int, early_stop_threshold: Optional[float]=4.0, progress_bar_refresh_rate: int=0, beta: float=0.98):
        self.num_training = num_training
        self.early_stop_threshold = early_stop_threshold
        self.beta = beta
        self.losses: List[float] = []
        self.lrs: List[float] = []
        self.avg_loss = 0.0
        self.best_loss = 0.0
        self.progress_bar_refresh_rate = progress_bar_refresh_rate
        self.progress_bar = None

    @override
    def on_train_batch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', batch: Any, batch_idx: int) -> None:
        """Called before each training batch, logs the lr that will be used."""
        if (trainer.fit_loop.batch_idx + 1) % trainer.accumulate_grad_batches != 0:
            return
        if self.progress_bar_refresh_rate and self.progress_bar is None:
            self.progress_bar = tqdm(desc='Finding best initial lr', total=self.num_training)
        self.lrs.append(trainer.lr_scheduler_configs[0].scheduler.lr[0])

    @override
    def on_train_batch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """Called when the training batch ends, logs the calculated loss."""
        if (trainer.fit_loop.batch_idx + 1) % trainer.accumulate_grad_batches != 0:
            return
        if not outputs:
            self.losses.append(float('nan'))
            return
        if self.progress_bar:
            self.progress_bar.update()
        loss_tensor = outputs if isinstance(outputs, torch.Tensor) else outputs['loss']
        assert loss_tensor is not None
        current_loss = loss_tensor.item()
        current_step = trainer.global_step
        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * current_loss
        smoothed_loss = self.avg_loss / (1 - self.beta ** (current_step + 1))
        if self.early_stop_threshold is not None and current_step > 1 and (smoothed_loss > self.early_stop_threshold * self.best_loss):
            trainer.should_stop = True
            if self.progress_bar:
                self.progress_bar.close()
        trainer.should_stop = trainer.strategy.broadcast(trainer.should_stop)
        if smoothed_loss < self.best_loss or current_step == 1:
            self.best_loss = smoothed_loss
        self.losses.append(smoothed_loss)