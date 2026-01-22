import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING
import numpy as np
import pyarrow as pa
from .. import config
from ..utils.py_utils import map_nested
from .formatting import TensorFormatter
def _tensorize(self, value):
    import torch
    if isinstance(value, (str, bytes, type(None))):
        return value
    elif isinstance(value, (np.character, np.ndarray)) and np.issubdtype(value.dtype, np.character):
        return value.tolist()
    default_dtype = {}
    if isinstance(value, (np.number, np.ndarray)) and np.issubdtype(value.dtype, np.integer):
        default_dtype = {'dtype': torch.int64}
        if value.dtype in [np.uint16, np.uint32]:
            value = value.astype(np.int64)
    elif isinstance(value, (np.number, np.ndarray)) and np.issubdtype(value.dtype, np.floating):
        default_dtype = {'dtype': torch.float32}
    elif config.PIL_AVAILABLE and 'PIL' in sys.modules:
        import PIL.Image
        if isinstance(value, PIL.Image.Image):
            value = np.asarray(value)
    return torch.tensor(value, **{**default_dtype, **self.torch_tensor_kwargs})