import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Union
from packaging import version
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name
from ..utils import is_accelerate_available, is_bitsandbytes_available, is_torch_available, logging
def adjust_target_dtype(self, target_dtype: 'torch.dtype') -> 'torch.dtype':
    if version.parse(importlib.metadata.version('accelerate')) > version.parse('0.19.0'):
        from accelerate.utils import CustomDtype
        if target_dtype != torch.int8:
            logger.info('target_dtype {target_dtype} is replaced by `CustomDtype.INT4` for 4-bit BnB quantization')
        return CustomDtype.INT4
    else:
        raise ValueError("You are using `device_map='auto'` on a 4bit loaded version of the model. To automatically compute the appropriate device map, you should upgrade your `accelerate` library,`pip install --upgrade accelerate` or install it from source to support fp4 auto device mapcalculation. You may encounter unexpected behavior, or pass your own device map")