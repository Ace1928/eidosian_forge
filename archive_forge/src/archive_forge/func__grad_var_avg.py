import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
import numpy as np
import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.optim import SGD, Optimizer
def _grad_var_avg(self, pg_idx: Optional[int]=None) -> float:
    """
        Current estimate of the trace of the covariance of the true gradient
        (mu squared in the AdaScale paper).

        Args:
            pg_idx (Optional[int]):
                Optional index for a parameter group.

        Returns:
            (float):
                Estimate of trace of the covariance.
        """
    if pg_idx is not None:
        return self._state['grad_var_avg'][pg_idx]
    else:
        return float(np.sum(self._state['grad_var_avg']))