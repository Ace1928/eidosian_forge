import abc
import torch
import itertools
import collections
from torch.nn.modules.module import _addindent
def _needs_weight_clamping(observer, dtype):
    observer = _get_weight_observer(observer)
    if dtype in [torch.qint8, torch.quint8, torch.qint32]:
        info = torch.iinfo(dtype)
        return observer.quant_min > info.min or observer.quant_max < info.max
    return False