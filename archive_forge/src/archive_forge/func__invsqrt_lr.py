from abc import abstractmethod
from torch import optim
import numpy as np
from parlai.core.exceptions import StopTrainException
from parlai.utils.misc import warn_once
def _invsqrt_lr(self, step):
    return self.decay_factor / np.sqrt(max(1, self.invsqrt_lr_decay_gamma + step))