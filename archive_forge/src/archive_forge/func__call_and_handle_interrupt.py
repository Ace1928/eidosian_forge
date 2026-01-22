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
def _call_and_handle_interrupt(trainer: 'pl.Trainer', trainer_fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Error handling, intended to be used only for main trainer function entry points (fit, validate, test, predict)
    as all errors should funnel through them.

    Args:
        trainer_fn: one of (fit, validate, test, predict)
        *args: positional arguments to be passed to the `trainer_fn`
        **kwargs: keyword arguments to be passed to `trainer_fn`

    """
    try:
        if trainer.strategy.launcher is not None:
            return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
        return trainer_fn(*args, **kwargs)
    except _TunerExitException:
        _call_teardown_hook(trainer)
        trainer._teardown()
        trainer.state.status = TrainerStatus.FINISHED
        trainer.state.stage = None
    except KeyboardInterrupt as exception:
        rank_zero_warn('Detected KeyboardInterrupt, attempting graceful shutdown...')
        if not trainer.interrupted:
            trainer.state.status = TrainerStatus.INTERRUPTED
            _call_callback_hooks(trainer, 'on_exception', exception)
            trainer.strategy.on_exception(exception)
            for logger in trainer.loggers:
                logger.finalize('failed')
    except BaseException as exception:
        trainer.state.status = TrainerStatus.INTERRUPTED
        _call_callback_hooks(trainer, 'on_exception', exception)
        trainer.strategy.on_exception(exception)
        for logger in trainer.loggers:
            logger.finalize('failed')
        trainer._teardown()
        trainer.state.stage = None
        raise