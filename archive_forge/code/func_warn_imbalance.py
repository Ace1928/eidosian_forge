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
def warn_imbalance(get_prop):
    values = [get_prop(props) for props in dev_props]
    min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
    max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
    if min_val / max_val < 0.75:
        warnings.warn(imbalance_warn.format(device_ids[min_pos], device_ids[max_pos]))
        return True
    return False