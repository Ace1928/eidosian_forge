from typing import TYPE_CHECKING, Optional
from .base import HfQuantizer
from ..integrations import replace_with_aqlm_linear
from ..utils import is_accelerate_available, is_aqlm_available, is_torch_available, logging
from ..utils.quantization_config import QuantizationConfigMixin

    Quantizer of the AQLM method. Enables the loading of prequantized models.
    