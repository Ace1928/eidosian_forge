import base64
import json
import os
from copy import deepcopy
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
class DeepSpeedSchedulerWrapper(AcceleratedScheduler):
    """
    Internal wrapper around a deepspeed scheduler.

    Args:
        scheduler (`torch.optim.lr_scheduler.LambdaLR`):
            The scheduler to wrap.
        optimizers (one or a list of `torch.optim.Optimizer`):
    """

    def __init__(self, scheduler, optimizers):
        super().__init__(scheduler, optimizers)

    def step(self):
        pass