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
def _get_output_act_obs_or_fq(arg: Node, named_modules: Dict[str, torch.nn.Module], obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize], is_qat: bool) -> ObserverOrFakeQuantize:
    """ Get the constructor for observer or fake quant object for
    the argument in the original graph as the output of previous node,
    skipping inserted observers

    We are assuming that the observers are inserted correctly, and the dtype for
    argument in quantized graph will match what is specified by the qconfig
    """
    assert isinstance(arg, Node)
    if 'quantization_annotation' in arg.meta:
        return _create_obs_or_fq_from_qspec(arg.meta['quantization_annotation'].output_qspec, obs_or_fq_map, is_qat)
    custom_module_lstm_node = _maybe_get_custom_module_lstm_from_node_arg(arg, named_modules)
    output_act_obs_or_fq_ctr = None
    if custom_module_lstm_node is not None:
        output_act_obs_or_fq_ctr = custom_module_lstm_node.meta['target_dtype_info']['output_act_obs_or_fq_ctr']
        output_act_obs_or_fq = output_act_obs_or_fq_ctr() if output_act_obs_or_fq_ctr else None
    elif _is_activation_post_process_node(arg, named_modules):
        observed_arg = arg.args[0]
        assert isinstance(observed_arg, Node), 'Currently we only support observing Node'
        if 'quantization_annotation' in observed_arg.meta:
            output_act_obs_or_fq = _create_obs_or_fq_from_qspec(observed_arg.meta['quantization_annotation'].output_qspec, obs_or_fq_map, is_qat)
        else:
            assert 'target_dtype_info' in observed_arg.meta
            output_act_obs_or_fq_ctr = observed_arg.meta['target_dtype_info']['output_act_obs_or_fq_ctr']
            output_act_obs_or_fq = output_act_obs_or_fq_ctr() if output_act_obs_or_fq_ctr else None
    else:
        if 'target_dtype_info' in arg.meta:
            output_act_obs_or_fq_ctr = arg.meta['target_dtype_info'].get('output_act_obs_or_fq_ctr', _DEFAULT_FP32_OBS_OR_FQ_CTR)
        else:
            output_act_obs_or_fq_ctr = _DEFAULT_FP32_OBS_OR_FQ_CTR
        output_act_obs_or_fq = output_act_obs_or_fq_ctr() if output_act_obs_or_fq_ctr else None
    return output_act_obs_or_fq