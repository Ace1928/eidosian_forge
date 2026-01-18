import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Union
from packaging import version
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name
from ..utils import is_accelerate_available, is_bitsandbytes_available, is_torch_available, logging
def update_torch_dtype(self, torch_dtype: 'torch.dtype') -> 'torch.dtype':
    if torch_dtype is None:
        logger.info('Overriding torch_dtype=%s with `torch_dtype=torch.float16` due to requirements of `bitsandbytes` to enable model loading in 8-bit or 4-bit. Pass your own torch_dtype to specify the dtype of the remaining non-linear layers or pass torch_dtype=torch.float16 to remove this warning.', torch_dtype)
        torch_dtype = torch.float16
    return torch_dtype