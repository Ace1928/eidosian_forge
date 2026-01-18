import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Union
from packaging import version
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name
from ..utils import is_accelerate_available, is_bitsandbytes_available, is_torch_available, logging

        combines logic from _load_state_dict_into_meta_model and .integrations.bitsandbytes.py::set_module_quantized_tensor_to_device()
        