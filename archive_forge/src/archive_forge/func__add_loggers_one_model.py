import collections
import torch
import torch.nn as nn
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.ao.ns.fx.mappings import (
from torch.ao.ns.fx.graph_matcher import (
from .fx.weight_utils import (
from .fx.graph_passes import (
from .fx.utils import (
from .fx.ns_types import (
from torch.ao.quantization.backend_config.utils import get_fusion_pattern_to_root_node_getter
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.fx.match_utils import _find_matches
from torch.ao.quantization.fx.graph_module import _get_observed_graph_module_attr
from torch.ao.quantization.fx.qconfig_mapping_utils import _generate_node_name_to_qconfig
from torch.ao.quantization.fx.quantize_handler import _get_pattern_to_quantize_handlers
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization import QConfigMapping
from torch.ao.ns.fx.n_shadows_utils import (
from torch.ao.ns.fx.qconfig_multi_mapping import QConfigMultiMapping
from typing import Dict, Tuple, Callable, List, Optional, Set, Any, Type
def _add_loggers_one_model(model_name: str, model: GraphModule, nodes_and_names_to_instrument_inputs: List[Tuple[Node, str, str]], nodes_and_names_to_instrument_outputs: List[Tuple[Node, str, str]], logger_cls: Callable) -> nn.Module:
    torch._C._log_api_usage_once('quantization_api._numeric_suite_fx._add_loggers_one_model')
    node_to_instrument_inputs_to_ref_name: Dict[Node, Tuple[str, str]] = {}
    node_to_instrument_outputs_to_ref_name: Dict[Node, Tuple[str, str]] = {}
    for node, ref_name, ref_node_type in nodes_and_names_to_instrument_inputs:
        node_to_instrument_inputs_to_ref_name[node] = (ref_name, ref_node_type)
    for node, ref_name, ref_node_type in nodes_and_names_to_instrument_outputs:
        node_to_instrument_outputs_to_ref_name[node] = (ref_name, ref_node_type)
    model = add_loggers_to_model(model, node_to_instrument_inputs_to_ref_name, node_to_instrument_outputs_to_ref_name, logger_cls, model_name)
    return model