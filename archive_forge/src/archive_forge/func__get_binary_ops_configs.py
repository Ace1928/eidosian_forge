import operator
from typing import List
import torch
import torch.ao.nn.qat as nnqat
import torch.ao.nn.quantized.reference as nnqr
import torch.nn as nn
import torch.nn.functional as F
from ..fuser_method_mappings import (
from ._common_operator_config_utils import _Conv2dMetadata
from .backend_config import (
from .qnnpack import (
def _get_binary_ops_configs() -> List[BackendPatternConfig]:
    """
    Return all configs related to binary ops.
    """
    dtype_configs = [qnnpack_default_op_qint8_symmetric_dtype_config, executorch_weighted_op_int8_dtype_config]
    num_tensor_args_to_observation_type_mapping = {0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT, 1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT, 2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT}
    binary_op_configs: List[BackendPatternConfig] = []
    for op in [operator.add, torch.add, operator.sub, torch.sub, operator.mul, torch.mul]:
        bop_patterns = [(op, torch.nn.ReLU), (op, torch.nn.functional.relu), (op, torch.relu), op]
        for bop_pattern in bop_patterns:
            binary_op_configs.append(BackendPatternConfig(bop_pattern).set_dtype_configs(dtype_configs)._set_num_tensor_args_to_observation_type(num_tensor_args_to_observation_type_mapping))
    return binary_op_configs