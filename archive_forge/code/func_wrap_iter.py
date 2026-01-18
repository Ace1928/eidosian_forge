import inspect
import functools
from enum import Enum
import torch.autograd
@functools.wraps(func)
def wrap_iter(*args, **kwargs):
    iter_ret = func(*args, **kwargs)
    datapipe = args[0]
    datapipe._snapshot_state = _SnapshotState.Iterating
    if datapipe._fast_forward_iterator:
        iter_ret = datapipe._fast_forward_iterator
        datapipe._fast_forward_iterator = None
        return iter_ret
    iterator_id = _set_datapipe_valid_iterator_id(datapipe)
    return IteratorDecorator(iter_ret, datapipe, iterator_id, '__next__' in namespace)