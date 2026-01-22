import os
import multiprocessing
from typing import TypeVar, Optional, Tuple, List
def _get_data_ptr(self, tensor: NumpyArrayOrPyTorchTensor):
    if self._is_pytorch_tensor(tensor):
        return tensor.data_ptr()
    else:
        return tensor.ctypes.data