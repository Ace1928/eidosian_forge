import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Union
from packaging import version
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name
from ..utils import is_accelerate_available, is_bitsandbytes_available, is_torch_available, logging
def update_device_map(self, device_map):
    if device_map is None:
        device_map = {'': torch.cuda.current_device()}
        logger.info("The device_map was not initialized. Setting device_map to {'':torch.cuda.current_device()}. If you want to use the model for inference, please set device_map ='auto' ")
    return device_map