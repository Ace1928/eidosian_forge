from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _coerce_types_torch(tensors):
    """Coerce a list of tensors to all have the same dtype
    without any loss of information."""
    torch = _i('torch')
    device_set = set((t.device for t in tensors if isinstance(t, torch.Tensor)))
    if len(device_set) > 1:
        device_names = ', '.join((str(d) for d in device_set))
        raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {device_names}!')
    device = device_set.pop() if len(device_set) == 1 else None
    tensors = [torch.as_tensor(t, device=device) for t in tensors]
    dtypes = {i.dtype for i in tensors}
    if len(dtypes) == 1:
        return tensors
    complex_priority = [torch.complex64, torch.complex128]
    float_priority = [torch.float16, torch.float32, torch.float64]
    int_priority = [torch.int8, torch.int16, torch.int32, torch.int64]
    complex_type = [i for i in complex_priority if i in dtypes]
    float_type = [i for i in float_priority if i in dtypes]
    int_type = [i for i in int_priority if i in dtypes]
    cast_type = complex_type or float_type or int_type
    cast_type = list(cast_type)[-1]
    return [t.to(cast_type) for t in tensors]