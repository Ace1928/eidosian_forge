from __future__ import annotations
import copy
import functools
from typing import Any, Callable, Dict, List, Optional, Set
import torch
import torch._dynamo as torchdynamo
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
from torch.fx import Node
def _get_supported_symmetric_config_and_operators() -> List[OperatorConfig]:
    supported_config_and_operators: List[OperatorConfig] = []
    for quantization_config in [get_symmetric_quantization_config(), get_symmetric_quantization_config(is_qat=True), get_symmetric_quantization_config(is_per_channel=True), get_symmetric_quantization_config(is_per_channel=True, is_qat=True)]:
        ops = _supported_symmetric_quantized_operators()
        for pattern_list in ops.values():
            supported_config_and_operators.append(OperatorConfig(quantization_config, pattern_list))
    return copy.deepcopy(supported_config_and_operators)