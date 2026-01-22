import collections
import dataclasses
import enum
import itertools as it
import logging
from typing import (
from typing_extensions import Literal
import torch
from torch._C import FunctionSchema
from torch._C._autograd import _ProfilerResult
from torch._C._profiler import (
from torch._utils import _element_size
from torch.profiler import _utils
def _coalesce_timeline(self, device_str):
    """Convert the memory timeline and categories into a memory plot
        consisting of timestamps and their respective sizes by category
        for a given device.

        Input: device
        Output: [timestamps, sizes by category]
        """
    device = torch.device(device_str)
    times: List[int] = []
    sizes: List[List[int]] = []

    def update(key, version, delta):
        category = self.categories.get(key, version) if isinstance(key, TensorKey) else None
        index = _CATEGORY_TO_INDEX[category] + 1
        sizes[-1][index] += int(delta)
    t_min = -1
    for t, action, (key, version), numbytes in self.timeline:
        if key.device != device:
            continue
        if t != -1:
            t = int(t / 1000)
        if t_min == -1 or (t < t_min and t > 0):
            t_min = t
        if len(times) == 0:
            times.append(t)
            sizes.append([0] + [0 for _ in _CATEGORY_TO_INDEX])
        elif t != times[-1]:
            times.append(t)
            sizes.append(sizes[-1].copy())
        if action in (Action.PREEXISTING, Action.CREATE):
            update(key, version, numbytes)
        elif action == Action.INCREMENT_VERSION:
            update(key, version, -numbytes)
            update(key, version + 1, numbytes)
        elif action == Action.DESTROY:
            update(key, version, -numbytes)
        else:
            raise ValueError(f'Unknown action: {action}')
    times = [t_min if t < 0 else t for t in times]
    return (times, sizes)