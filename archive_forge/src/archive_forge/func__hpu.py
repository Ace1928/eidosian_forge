import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def _hpu(self, device=None, non_blocking=False, **kwargs):
    """Returns a copy of this object in HPU memory.

    If this object is already in HPU memory and on the correct device, then
    no copy is performed and the original object is returned.

    Args:
        device (int): The destination HPU id. Defaults to the current device.
        non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
        **kwargs: For compatibility, may contain the key ``async`` in place of
            the ``non_blocking`` argument.
    """
    non_blocking = _get_async_or_non_blocking('hpu', non_blocking, kwargs)
    hpu = getattr(torch, 'hpu', None)
    assert hpu is not None, 'HPU device module is not loaded'
    if self.is_hpu:
        if device is None:
            device = hpu.current_device()
        if self.get_device() == device:
            return self
    elif device is None:
        device = -1
    with hpu.device(device):
        assert not self.is_sparse, 'sparse storage is not supported for HPU tensors'
        untyped_storage = torch.UntypedStorage(self.size(), device=torch.device('hpu'))
        untyped_storage.copy_(self, non_blocking)
        return untyped_storage