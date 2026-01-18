import importlib.metadata
from typing import TYPE_CHECKING
from packaging import version
from .base import HfQuantizer
from ..utils import is_accelerate_available, is_auto_awq_available, is_torch_available, logging

    4-bit quantization for Activation-aware Weight Quantization(AWQ) (https://arxiv.org/abs/2306.00978)
    