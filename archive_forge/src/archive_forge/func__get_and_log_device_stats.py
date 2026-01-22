from typing import Any, Dict, Optional
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.accelerators.cpu import _PSUTIL_AVAILABLE
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT
def _get_and_log_device_stats(self, trainer: 'pl.Trainer', key: str) -> None:
    if not trainer._logger_connector.should_update_logs:
        return
    device = trainer.strategy.root_device
    if self._cpu_stats is False and device.type == 'cpu':
        return
    device_stats = trainer.accelerator.get_device_stats(device)
    if self._cpu_stats and device.type != 'cpu':
        from pytorch_lightning.accelerators.cpu import get_cpu_stats
        device_stats.update(get_cpu_stats())
    for logger in trainer.loggers:
        separator = logger.group_separator
        prefixed_device_stats = _prefix_metric_keys(device_stats, f'{self.__class__.__qualname__}.{key}', separator)
        logger.log_metrics(prefixed_device_stats, step=trainer.fit_loop.epoch_loop._batches_that_stepped)