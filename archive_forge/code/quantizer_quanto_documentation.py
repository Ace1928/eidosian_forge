import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from packaging import version
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name
from ..utils import is_accelerate_available, is_quanto_available, is_torch_available, logging
from ..utils.quantization_config import QuantoConfig

        Create the quantized parameter by calling .freeze() after setting it to the module.
        