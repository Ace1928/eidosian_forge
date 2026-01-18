import itertools
from warnings import warn
import torch
import torch.cuda
from torch.autograd import (
from torch.autograd.profiler_util import (
def total_average(self):
    self._check_finish()
    assert self.function_events is not None, 'Expected profiling results'
    return self.function_events.total_average()