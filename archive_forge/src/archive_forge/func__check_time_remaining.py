import logging
import time
from datetime import timedelta
from typing import Any, Dict, Optional, Union
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_info
def _check_time_remaining(self, trainer: 'pl.Trainer') -> None:
    assert self._duration is not None
    should_stop = self.time_elapsed() >= self._duration
    should_stop = trainer.strategy.broadcast(should_stop)
    trainer.should_stop = trainer.should_stop or should_stop
    if should_stop and self._verbose:
        elapsed = timedelta(seconds=int(self.time_elapsed(RunningStage.TRAINING)))
        rank_zero_info(f'Time limit reached. Elapsed time is {elapsed}. Signaling Trainer to stop.')