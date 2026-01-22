import torch
import re
from collections import defaultdict, OrderedDict
from typing import Callable, Any, Dict, Tuple, Set, List, Union
from torch.ao.quantization import QConfig
from torch.ao.quantization.qconfig import _add_module_to_qconfig_obs_ctr, QConfigAny, qconfig_equals
from torch.ao.quantization.observer import (
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.backend_config.utils import (
from torch.fx import (
from torch.fx.graph import (
from torch.ao.nn.intrinsic import _FusedModule
from ..utils import (
from ..qconfig_mapping import (
def _update_qconfig_for_fusion(model: GraphModule, qconfig_mapping: QConfigMapping):
    """
    Update the QConfigMapping to account for fused modules such as LinearReLU.
    This assumes the QConfigMapping's attributes have already been converted to OrderedDicts.
    """
    object_type_dict = qconfig_mapping.object_type_qconfigs
    if len(object_type_dict) == 0:
        return qconfig_mapping
    modules = dict(model.named_modules())
    for node in model.graph.nodes:
        if node.op == 'call_module' and node.target in modules:
            maybe_fused_module = modules[str(node.target)]
            if not isinstance(maybe_fused_module, _FusedModule):
                continue
            ops = list(maybe_fused_module._modules.values())
            fused_qconfig = object_type_dict.get(type(ops[0]), None)
            for op in ops[1:]:
                if not qconfig_equals(object_type_dict.get(type(op), None), fused_qconfig):
                    raise LookupError('During fusion, we need to specify the same ' + f'qconfigs for all module types in {type(maybe_fused_module)} ' + f'offending type: {type(op)}')
            if fused_qconfig is not None:
                object_type_dict[type(maybe_fused_module)] = fused_qconfig