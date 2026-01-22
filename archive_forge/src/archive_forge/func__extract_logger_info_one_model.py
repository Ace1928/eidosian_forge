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
def _extract_logger_info_one_model(model: nn.Module, results: NSResultsType, logger_cls: Callable) -> None:
    torch._C._log_api_usage_once('quantization_api._numeric_suite_fx._extract_logger_info_one_model')
    for gm_name, mod in model.named_modules():
        is_logger = isinstance(mod, logger_cls) or (isinstance(mod, torch.jit.RecursiveScriptModule) and mod.original_name == 'OutputLogger')
        if is_logger:
            key = mod.ref_name
            if key not in results:
                results[key] = {}
            assert mod.model_name not in results[key], f'{mod.model_name} is already present in results'
            if mod.results_type not in results[key]:
                results[key][mod.results_type] = {}
            if mod.model_name not in results[key][mod.results_type]:
                results[key][mod.results_type][mod.model_name] = []
            stats_to_use = mod.stats
            if len(mod.stats_rnn) > 0:
                stats_to_use = mod.stats_rnn
            data = {'type': mod.results_type, 'values': stats_to_use, 'ref_node_name': mod.ref_node_name, 'ref_node_target_type': mod.ref_node_target_type, 'prev_node_name': mod.prev_node_name, 'prev_node_target_type': mod.prev_node_target_type, 'index_within_arg': mod.index_within_arg, 'index_of_arg': mod.index_of_arg, 'fqn': mod.fqn, 'qconfig_str': mod.qconfig_str}
            if hasattr(mod, 'comparisons'):
                data['comparisons'] = mod.comparisons
                data['comparison_fn_name'] = mod.comparison_fn_name
            else:
                data['comparisons'] = []
                data['comparison_fn_name'] = ''
            results[key][mod.results_type][mod.model_name].append(data)
            results[key][mod.results_type][mod.model_name].sort(key=lambda res: f'{res['index_of_arg']}:{res['index_within_arg']}')