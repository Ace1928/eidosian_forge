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
def _is_pattern_dtype_config_and_qconfig_supported_by_backend(pattern: Optional[Pattern], matched_node_pattern: Optional[List[Node]], qconfig: QConfigAny, backend_config: BackendConfig) -> bool:
    """ Check if the dtype configuration of a pattern is supported by
    the backend or not, and whether the qconfig satisfies constraints
    specified in the corresponding dtype config.
    """
    if backend_config is None or pattern is None:
        return True
    assert matched_node_pattern is not None and len(matched_node_pattern) >= 1
    pattern_to_dtype_configs = get_pattern_to_dtype_configs(backend_config)
    dtype_configs: List[DTypeConfig] = pattern_to_dtype_configs.get(pattern, [])
    pattern_to_root_node_getter = get_fusion_pattern_to_root_node_getter(backend_config)
    root_node_getter = pattern_to_root_node_getter.get(pattern, _default_root_node_getter)
    root_node = root_node_getter(matched_node_pattern)
    input_node = root_node
    output_node = matched_node_pattern[0]
    for dtype_config in dtype_configs:
        supported = True
        for arg in list(input_node.args) + list(input_node.kwargs.values()):
            supported = supported and _is_input_arg_dtype_supported_by_backend(arg, input_node, qconfig, dtype_config, backend_config)
        supported = supported and _is_output_dtype_supported_by_backend(output_node, qconfig, dtype_config)
        if supported:
            return True
    return False