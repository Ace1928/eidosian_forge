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
def _is_output_dtype_supported_by_backend(node: Node, qconfig: QConfigAny, dtype_config: DTypeConfig) -> bool:
    """ Check if the configured qconfig for the output
    is supported by the backend or not
    """
    backend_config_output_dtype = dtype_config.output_dtype
    qconfig_output_dtype = None
    output_act_obs_or_fq_ctr = node.meta['target_dtype_info'].get('output_act_obs_or_fq_ctr', _DEFAULT_FP32_OBS_OR_FQ_CTR)
    output_act_obs_or_fq = output_act_obs_or_fq_ctr() if output_act_obs_or_fq_ctr else None
    qconfig_output_dtype, qconfig_output_is_dynamic = _get_dtype_and_is_dynamic(output_act_obs_or_fq)
    if qconfig_output_is_dynamic:
        qconfig_output_dtype = torch.float32
    dtype_matches = qconfig_output_dtype == backend_config_output_dtype
    qconfig_satisfies_constraints = _qconfig_satisfies_dtype_config_constraints(qconfig, dtype_config.output_dtype_with_constraints)
    return backend_config_output_dtype is None or (dtype_matches and qconfig_satisfies_constraints)