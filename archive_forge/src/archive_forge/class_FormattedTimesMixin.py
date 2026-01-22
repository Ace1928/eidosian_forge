import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
class FormattedTimesMixin:
    """Helpers for FunctionEvent and FunctionEventAvg.

    The subclass should define `*_time_total` and `count` attributes.
    """
    cpu_time_str = _attr_formatter('cpu_time')
    cuda_time_str = _attr_formatter('cuda_time')
    privateuse1_time_str = _attr_formatter('privateuse1_time')
    cpu_time_total_str = _attr_formatter('cpu_time_total')
    cuda_time_total_str = _attr_formatter('cuda_time_total')
    privateuse1_time_total_str = _attr_formatter('privateuse1_time_total')
    self_cpu_time_total_str = _attr_formatter('self_cpu_time_total')
    self_cuda_time_total_str = _attr_formatter('self_cuda_time_total')
    self_privateuse1_time_total_str = _attr_formatter('self_privateuse1_time_total')

    @property
    def cpu_time(self):
        return 0.0 if self.count == 0 else 1.0 * self.cpu_time_total / self.count

    @property
    def cuda_time(self):
        return 0.0 if self.count == 0 else 1.0 * self.cuda_time_total / self.count

    @property
    def privateuse1_time(self):
        return 0.0 if self.count == 0 else 1.0 * self.privateuse1_time_total / self.count