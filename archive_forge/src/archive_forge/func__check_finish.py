import itertools
from warnings import warn
import torch
import torch.cuda
from torch.autograd import (
from torch.autograd.profiler_util import (
def _check_finish(self):
    if self.function_events is None:
        raise RuntimeError("Profiler didn't finish running")