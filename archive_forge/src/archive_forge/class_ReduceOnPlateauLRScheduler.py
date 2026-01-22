from abc import abstractmethod
from torch import optim
import numpy as np
from parlai.core.exceptions import StopTrainException
from parlai.utils.misc import warn_once
class ReduceOnPlateauLRScheduler(ParlAILRScheduler):
    """
    Scheduler that decays by a multiplicative rate when valid loss plateaus.
    """

    def __init__(self, optimizer, hard_reset, patience, decay, warmup_updates, warmup_rate):
        super().__init__(hard_reset, warmup_updates, warmup_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=decay, patience=patience, verbose=True)

    def train_step(self, scheduler_steps):
        pass

    def valid_step(self, metrics_dict):
        if self._is_lr_warming_up():
            return
        if 'loss' not in metrics_dict:
            warn_once("LR scheduler expected to see loss metric, but didn't.")
            return
        self.scheduler.step(metrics_dict['loss'])