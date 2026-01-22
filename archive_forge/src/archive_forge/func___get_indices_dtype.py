import warnings
from collections import namedtuple
from typing import Any, Optional
import torch
@staticmethod
def __get_indices_dtype(values_dtype):
    if values_dtype == torch.int8:
        return torch.int32
    elif values_dtype in (torch.float16, torch.bfloat16):
        return torch.int16
    else:
        raise RuntimeError(f'Datatype {values_dtype}  is not supported!')
    return None