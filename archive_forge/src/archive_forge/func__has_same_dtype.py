import torch
from torch._subclasses import FakeTensor
from torch.ao.quantization.fx.prepare import (
from torch.fx import (
from torch.fx.node import Argument
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
from typing import Dict, Tuple, Union, Any, Optional
from torch.ao.quantization.quantizer import (
from torch.ao.quantization import ObserverOrFakeQuantize
def _has_same_dtype(qspec_a: QuantizationSpecBase, qspec_b: QuantizationSpecBase):
    return hasattr(qspec_a, 'dtype') and hasattr(qspec_b, 'dtype') and (qspec_a.dtype == qspec_b.dtype)