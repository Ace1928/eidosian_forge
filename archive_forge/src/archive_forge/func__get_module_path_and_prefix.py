from typing import Any, Dict, List, Optional, Set, Tuple, Union, Type, Callable
from torch.ao.quantization.quant_type import QuantType
import torch
import copy
import warnings
from torch.fx import (
from torch.fx.graph import (
from ..utils import (
from ..qconfig import (
from ..qconfig_mapping import QConfigMapping
from .qconfig_mapping_utils import (
from torch.ao.quantization.backend_config.utils import (
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.observer import _is_activation_post_process
from .graph_module import (
from ._equalize import update_obs_for_equalization, convert_eq_obs
from torch.nn.utils.parametrize import type_before_parametrizations
from .utils import (
from torch.ao.quantization.utils import (
from torch.ao.quantization.quantize import (
from torch.ao.quantization.stubs import DeQuantStub
from .custom_config import (
from .lower_to_fbgemm import lower_to_fbgemm
from ._decomposed import quantized_decomposed_lib  # noqa: F401
import operator
def _get_module_path_and_prefix(obs_node: Node, node_name_to_scope: Dict[str, Tuple[str, type]], node_name_to_qconfig: Dict[str, QConfigAny]) -> Tuple[str, str]:
    """ Given and observer node, get the `Scope` or the fully qualified name for
    the submodule containing the observed node, also return a prefix of "_input"
    when the observed node is an input of a F.linear op, and not the output of another
    quantized op.
    TODO: this logic is hacky, we should think about how to remove it or make it more
    general
    """
    observed_node = obs_node.args[0]
    assert isinstance(observed_node, Node), f'Expecting observed node to be a Node, but got {observed_node}'
    is_input_observer_only = node_name_to_qconfig[observed_node.name] is None if observed_node.name in node_name_to_qconfig else None
    if is_input_observer_only:
        users = list(obs_node.users)
        first_linear_use_or_first_use = users[0] if users else None
        linear_node = None
        for n in users:
            if n.op == 'call_function' and n.target == torch.nn.functional.linear:
                linear_node = n
                break
        if linear_node:
            first_linear_use_or_first_use = linear_node
        prefix = '_input'
    else:
        first_linear_use_or_first_use = observed_node
        prefix = ''
    if first_linear_use_or_first_use and first_linear_use_or_first_use.name in node_name_to_scope:
        module_path, _ = node_name_to_scope[first_linear_use_or_first_use.name]
    else:
        module_path = ''
    return (module_path, prefix)