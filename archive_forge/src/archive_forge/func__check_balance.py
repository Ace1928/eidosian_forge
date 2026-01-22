import operator
import torch
import warnings
from itertools import chain
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar, Union
from ..modules import Module
from .scatter_gather import scatter_kwargs, gather
from .replicate import replicate
from .parallel_apply import parallel_apply
from torch._utils import (
def _check_balance(device_ids: Sequence[Union[int, torch.device]]) -> None:
    imbalance_warn = '\n    There is an imbalance between your GPUs. You may want to exclude GPU {} which\n    has less than 75% of the memory or cores of GPU {}. You can do so by setting\n    the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES\n    environment variable.'
    device_ids = [_get_device_index(x, True) for x in device_ids]
    dev_props = _get_devices_properties(device_ids)

    def warn_imbalance(get_prop):
        values = [get_prop(props) for props in dev_props]
        min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
        max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
        if min_val / max_val < 0.75:
            warnings.warn(imbalance_warn.format(device_ids[min_pos], device_ids[max_pos]))
            return True
        return False
    if warn_imbalance(lambda props: props.total_memory):
        return
    if warn_imbalance(lambda props: props.multi_processor_count):
        return