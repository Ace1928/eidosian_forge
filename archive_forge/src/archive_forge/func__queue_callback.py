import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
import numpy as np
import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.optim import SGD, Optimizer
def _queue_callback(self) -> None:
    if self._final_callback_queued:
        return
    self._final_callback_queued = True
    Variable._execution_engine.queue_callback(self._final_callback)