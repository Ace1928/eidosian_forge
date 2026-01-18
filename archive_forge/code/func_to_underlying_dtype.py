import functools
import warnings
from collections import OrderedDict
from inspect import getfullargspec, signature
from typing import Any, Callable, Dict, Optional, Tuple, Union
import torch
from torch.ao.quantization.quant_type import QuantType
from torch.fx import Node
from torch.nn.utils.parametrize import is_parametrized
def to_underlying_dtype(qdtype):
    DTYPE_MAPPING = {torch.quint8: torch.uint8, torch.qint8: torch.int8, torch.qint32: torch.int32, torch.quint4x2: torch.uint8, torch.quint2x4: torch.uint8, torch.uint8: torch.uint8, torch.int8: torch.int8, torch.int16: torch.int16, torch.int32: torch.int32}
    assert qdtype in DTYPE_MAPPING, 'Unsupported dtype: ' + qdtype
    return DTYPE_MAPPING[qdtype]