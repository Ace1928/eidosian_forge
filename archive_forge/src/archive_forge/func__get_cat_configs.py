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
def _get_cat_configs() -> List[BackendPatternConfig]:
    dtype_configs = [qnnpack_default_op_qint8_symmetric_dtype_config, executorch_default_op_quint8_dtype_config]
    cat_configs = []
    cat_configs.append(BackendPatternConfig(torch.cat).set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT).set_dtype_configs(dtype_configs))
    cat_configs.append(BackendPatternConfig(torch.concat).set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT).set_dtype_configs(dtype_configs))
    cat_configs.append(BackendPatternConfig(torch.concatenate).set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT).set_dtype_configs(dtype_configs))
    return cat_configs