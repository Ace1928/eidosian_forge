import logging
import os
import uuid
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple
import pytorch_lightning as pl
from pytorch_lightning.utilities.memory import garbage_collection_cuda, is_oom_error
from pytorch_lightning.utilities.parsing import lightning_getattr, lightning_setattr
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
def __scale_batch_restore_params(trainer: 'pl.Trainer', params: Dict[str, Any]) -> None:
    trainer.loggers = params['loggers']
    trainer.callbacks = params['callbacks']
    loop = trainer._active_loop
    assert loop is not None
    if isinstance(loop, pl.loops._FitLoop):
        loop.epoch_loop.max_steps = params['max_steps']
        trainer.limit_train_batches = params['limit_train_batches']
        trainer.limit_val_batches = params['limit_val_batches']
    elif isinstance(loop, pl.loops._EvaluationLoop):
        stage = trainer.state.stage
        assert stage is not None
        setattr(trainer, f'limit_{stage.dataloader_prefix}_batches', params['limit_eval_batches'])
    loop.load_state_dict(deepcopy(params['loop_state_dict']))
    loop.restarting = False
    if isinstance(loop, pl.loops._EvaluationLoop) and 'loop_verbose' in params:
        loop.verbose = params['loop_verbose']
    _reset_dataloaders(trainer)
    loop.reset()