import logging
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Type, Union
from packaging.version import Version
import pytorch_lightning as pl
from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from pytorch_lightning.callbacks import Checkpoint, EarlyStopping
from pytorch_lightning.trainer.states import TrainerStatus
from pytorch_lightning.utilities.exceptions import _TunerExitException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def _call_teardown_hook(trainer: 'pl.Trainer') -> None:
    assert trainer.state.fn is not None
    fn = trainer.state.fn
    if trainer.datamodule is not None:
        _call_lightning_datamodule_hook(trainer, 'teardown', stage=fn)
    _call_callback_hooks(trainer, 'teardown', stage=fn)
    _call_lightning_module_hook(trainer, 'teardown', stage=fn)
    trainer.lightning_module._current_fx_name = None
    trainer.lightning_module._metric_attributes = None
    for logger in trainer.loggers:
        logger.finalize('success')
    trainer.profiler.describe()