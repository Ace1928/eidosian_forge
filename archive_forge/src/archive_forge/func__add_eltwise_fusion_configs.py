import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import torch.nn.functional as F
import torch.ao.nn.quantized.reference as nnqr
from ._common_operator_config_utils import (
from .backend_config import (
from ..fuser_method_mappings import (
import operator
from torch.ao.quantization.utils import MatchAllNode
import itertools
def _add_eltwise_fusion_configs(configs, root_module, root_op, post_module, post_op, dtype_configs, fuser_method, fused_module, observation_type, ref_quant_module):
    configs.append(BackendPatternConfig((root_module, post_module)).set_dtype_configs(dtype_configs).set_fuser_method(fuser_method).set_fused_module(fused_module))
    configs.append(BackendPatternConfig((root_module, post_op)).set_dtype_configs(dtype_configs).set_fuser_method(fuser_method).set_fused_module(fused_module))
    configs.append(BackendPatternConfig(fused_module).set_observation_type(observation_type).set_dtype_configs(dtype_configs).set_root_module(root_module).set_reference_quantized_module(ref_quant_module))
    configs.append(BackendPatternConfig((root_op, post_module)).set_observation_type(observation_type).set_dtype_configs(dtype_configs))
    configs.append(BackendPatternConfig((root_op, post_op)).set_observation_type(observation_type).set_dtype_configs(dtype_configs))