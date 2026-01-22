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
def _run_prepare_fx_on_standalone_modules(model: torch.nn.Module, is_qat: bool, named_modules: Dict[str, torch.nn.Module], node_name_to_match_result_with_qconfig: Any, prepare_custom_config: PrepareCustomConfig, backend_config: BackendConfig) -> None:
    """
    Runs prepare_fx on each standalone module. Note: this does
    not modify the graph, it just replaces the unobserved modules with
    their observed versions.
    """
    for root_node, _, pattern, qhandler, qconfig in node_name_to_match_result_with_qconfig.values():
        if qhandler is None:
            continue
        elif not qhandler.is_standalone_module():
            continue
        sm_qconfig_mapping, sm_example_inputs, sm_prepare_custom_config, sm_backend_config = _get_standalone_module_configs(root_node, named_modules, prepare_custom_config, qconfig, backend_config)
        standalone_module = named_modules[root_node.target]
        prepare = torch.ao.quantization.quantize_fx._prepare_standalone_module_fx
        observed_standalone_module = prepare(standalone_module, sm_qconfig_mapping, is_qat, example_inputs=sm_example_inputs, prepare_custom_config=sm_prepare_custom_config, backend_config=sm_backend_config)
        parent_name, name = _parent_name(root_node.target)
        setattr(named_modules[parent_name], name, observed_standalone_module)
        named_modules[root_node.target] = observed_standalone_module