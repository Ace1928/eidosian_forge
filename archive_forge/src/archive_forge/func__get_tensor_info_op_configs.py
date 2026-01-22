import copy
import operator
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.qat as nnqat
import torch.ao.nn.quantized.reference as nnqr
from collections import namedtuple
from typing import Callable, Dict, List, Union
from .backend_config import (
from ..fuser_method_mappings import (
def _get_tensor_info_op_configs(dtype_configs):
    """
    These ops work on tensors of different dtypes but return non-tensors
    containing information about the input tensor.
    """

    def _get_config(op):
        return BackendPatternConfig(op).set_observation_type(ObservationType.INPUT_OUTPUT_NOT_OBSERVED).set_dtype_configs(dtype_configs)
    return [_get_config(op) for op in ('shape', 'size')]