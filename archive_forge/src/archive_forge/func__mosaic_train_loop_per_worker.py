import inspect
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type
from composer.loggers.logger_destination import LoggerDestination
from composer.trainer import Trainer
from ray.train import Checkpoint, DataConfig, RunConfig, ScalingConfig
from ray.train.mosaic._mosaic_utils import RayLogger
from ray.train.torch import TorchConfig, TorchTrainer
from ray.train.trainer import GenDataset
from ray.util import PublicAPI
def _mosaic_train_loop_per_worker(config):
    """Per-worker training loop for Mosaic Composers."""
    trainer_init_per_worker = config.pop('_trainer_init_per_worker')
    ray_logger = RayLogger(keys=config.pop('log_keys', []))
    trainer: Trainer = trainer_init_per_worker(config)
    filtered_callbacks = list()
    for callback in trainer.state.callbacks:
        if not isinstance(callback, LoggerDestination):
            filtered_callbacks.append(callback)
    filtered_callbacks.append(ray_logger)
    trainer.state.callbacks = filtered_callbacks
    trainer.logger.destinations = (ray_logger,)
    trainer.fit()