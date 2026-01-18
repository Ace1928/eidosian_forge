import copy
import io
from typing import List, Union
import torch
def torch__util__rebuild_tensor_wrapper(storage, storage_offset, size, stride):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.utils._mode_utils import no_dispatch
    from torch.utils._python_dispatch import _get_current_dispatch_mode

    def _rebuild_real_tensor(storage, storage_offset, size, stride):
        t = torch.tensor([], dtype=storage.dtype, device=storage._untyped_storage.device)
        return t.set_(storage._untyped_storage, storage_offset, size, stride)
    mode = _get_current_dispatch_mode()
    if isinstance(mode, FakeTensorMode):
        with no_dispatch():
            t = _rebuild_real_tensor(storage, storage_offset, size, stride)
        return mode.from_tensor(t)
    return _rebuild_real_tensor(storage, storage_offset, size, stride)