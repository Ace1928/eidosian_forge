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
def _lr_find(trainer: 'pl.Trainer', model: 'pl.LightningModule', min_lr: float=1e-08, max_lr: float=1, num_training: int=100, mode: str='exponential', early_stop_threshold: Optional[float]=4.0, update_attr: bool=False, attr_name: str='') -> Optional[_LRFinder]:
    """Enables the user to do a range test of good initial learning rates, to reduce the amount of guesswork in picking
    a good starting learning rate.

    Args:
        trainer: A Trainer instance.
        model: Model to tune.
        min_lr: minimum learning rate to investigate
        max_lr: maximum learning rate to investigate
        num_training: number of learning rates to test
        mode: Search strategy to update learning rate after each batch:

            - ``'exponential'``: Increases the learning rate exponentially.
            - ``'linear'``: Increases the learning rate linearly.

        early_stop_threshold: Threshold for stopping the search. If the
            loss at any point is larger than early_stop_threshold*best_loss
            then the search is stopped. To disable, set to None.
        update_attr: Whether to update the learning rate attribute or not.
        attr_name: Name of the attribute which stores the learning rate. The names 'learning_rate' or 'lr' get
            automatically detected. Otherwise, set the name here.

    """
    if trainer.fast_dev_run:
        rank_zero_warn('Skipping learning rate finder since `fast_dev_run` is enabled.')
        return None
    if update_attr:
        attr_name = _determine_lr_attr_name(model, attr_name)
    ckpt_path = os.path.join(trainer.default_root_dir, f'.lr_find_{uuid.uuid4()}.ckpt')
    ckpt_path = trainer.strategy.broadcast(ckpt_path)
    trainer.save_checkpoint(ckpt_path)
    start_steps = trainer.global_step
    params = __lr_finder_dump_params(trainer)
    __lr_finder_reset_params(trainer, num_training, early_stop_threshold)
    if trainer.progress_bar_callback:
        trainer.progress_bar_callback.disable()
    lr_finder = _LRFinder(mode, min_lr, max_lr, num_training)
    lr_finder._exchange_scheduler(trainer)
    _try_loop_run(trainer, params)
    if trainer.global_step != num_training + start_steps:
        log.info(f'LR finder stopped early after {trainer.global_step} steps due to diverging loss.')
    lr_finder.results.update({'lr': trainer.callbacks[0].lrs, 'loss': trainer.callbacks[0].losses})
    lr_finder._total_batch_idx = trainer.fit_loop.total_batch_idx
    __lr_finder_restore_params(trainer, params)
    if trainer.progress_bar_callback:
        trainer.progress_bar_callback.enable()
    lr_finder.results = trainer.strategy.broadcast(lr_finder.results)
    if update_attr:
        lr = lr_finder.suggestion()
        if lr is not None:
            lightning_setattr(model, attr_name, lr)
            log.info(f'Learning rate set to {lr}')
    trainer._checkpoint_connector.restore(ckpt_path)
    trainer.strategy.remove_checkpoint(ckpt_path)
    trainer.fit_loop.restarting = False
    trainer.fit_loop.epoch_loop.val_loop._combined_loader = None
    return lr_finder