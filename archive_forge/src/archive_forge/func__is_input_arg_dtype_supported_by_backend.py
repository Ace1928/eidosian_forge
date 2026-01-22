import copy
import torch
import warnings
from torch.fx import (
from torch.fx.graph import (
from torch.fx.node import Argument
from ..quantize import (
from ..observer import (
from ..qconfig import (
from ..qconfig_mapping import (
from .qconfig_mapping_utils import (
from .quantize_handler import (
from torch.ao.quantization import (
from torch.ao.quantization.utils import (
from ._equalize import (
from .pattern_utils import (
from .match_utils import (
from .utils import (
from torch.ao.quantization import (
from torch.ao.quantization.quantize import (
from ..utils import (
from ..backend_config.utils import (
from ..backend_config import (
from .custom_config import (
from torch.ao.quantization.quantizer import (
from torch.ao.quantization import ObserverOrFakeQuantize
from torch._subclasses import FakeTensor
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from dataclasses import asdict
def _is_input_arg_dtype_supported_by_backend(arg: Argument, node: Node, qconfig: QConfigAny, dtype_config: DTypeConfig, backend_config: BackendConfig) -> bool:
    """ Check if the configured qconfig for the argument
    is supported by the backend or not
    """
    if isinstance(arg, (list, tuple)):
        return all((_is_input_arg_dtype_supported_by_backend(a, node, qconfig, dtype_config, backend_config) for a in arg))
    if not isinstance(arg, Node):
        return True
    is_weight = node_arg_is_weight(node, arg)
    is_bias = node_arg_is_bias(node, arg)
    is_activation = not is_weight and (not is_bias)
    if is_activation:
        input_act_obs_or_fq_ctr = node.meta['target_dtype_info'].get('input_act_obs_or_fq_ctr')
        input_act_obs_or_fq = input_act_obs_or_fq_ctr() if input_act_obs_or_fq_ctr else None
        qconfig_dtype, qconfig_is_dynamic = _get_dtype_and_is_dynamic(input_act_obs_or_fq)
        return dtype_config.input_dtype is None or (dtype_config.input_dtype == qconfig_dtype and bool(dtype_config.is_dynamic) == bool(qconfig_is_dynamic) and _qconfig_satisfies_dtype_config_constraints(qconfig, dtype_config.input_dtype_with_constraints))
    elif is_weight:
        weight_obs_or_fq_ctr = node.meta['target_dtype_info'].get('weight_obs_or_fq_ctr', None)
        weight_obs_or_fq = weight_obs_or_fq_ctr() if weight_obs_or_fq_ctr else None
        qconfig_weight_dtype, _ = _get_dtype_and_is_dynamic(weight_obs_or_fq)
        backend_config_weight_dtype = dtype_config.weight_dtype
        dtype_matches = qconfig_weight_dtype == backend_config_weight_dtype
        qconfig_satisfies_constraints = _qconfig_satisfies_dtype_config_constraints(qconfig, dtype_config.weight_dtype_with_constraints, is_activation=False)
        return backend_config_weight_dtype is None or (dtype_matches and qconfig_satisfies_constraints)
    else:
        bias_obs_or_fq_ctr = node.meta['target_dtype_info'].get('bias_obs_or_fq_ctr', None)
        bias_obs_or_fq = bias_obs_or_fq_ctr() if bias_obs_or_fq_ctr else None
        qconfig_bias_dtype, _ = _get_dtype_and_is_dynamic(bias_obs_or_fq)
        backend_config_bias_dtype = dtype_config.bias_dtype
        return backend_config_bias_dtype is None or qconfig_bias_dtype == backend_config_bias_dtype