import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
import numpy as np
import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.optim import SGD, Optimizer
def _update_avg(self, name: str, value: np.ndarray, factor: float) -> None:
    if self._debias_ewma:
        biased = self._state.get(name + '_biased', np.zeros(value.shape[0]))
        unbias = self._state.get(name + '_unbias', np.zeros(value.shape[0]))
        biased = factor * biased + (1.0 - factor) * value
        unbias = factor * unbias + (1.0 - factor)
        self._state[name + '_biased'] = biased
        self._state[name + '_unbias'] = unbias
        self._state[name] = biased / unbias
    else:
        count = self._state.get(name + '_count', np.zeros(1))
        count[0] += 1
        self._state[name + '_count'] = count
        if count < 1 / (1 - self._smoothing):
            total = self._state.get(name + '_total', None)
            if total is None:
                total = value
            else:
                total += value
            self._state[name + '_total'] = total
            self._state[name] = total / count
        else:
            self._state[name] = factor * self._state[name] + (1.0 - factor) * value