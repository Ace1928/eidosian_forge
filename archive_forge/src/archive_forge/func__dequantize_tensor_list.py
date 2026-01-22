import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
from torch.ao.quantization import prepare
from typing import Dict, List, Optional, Any, Union, Callable, Set
from torch.ao.quantization.quantization_mappings import (
def _dequantize_tensor_list(t: Any) -> Any:
    return [_dequantize_tensor_list(x) for x in t] if type(t) is list else t.dequantize() if t.is_quantized else t