import warnings
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic as nni
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.ao.quantization.fx.graph_module import _get_observed_graph_module_attr
from ..observer import _with_args, ObserverBase, PerChannelMinMaxObserver
from ..utils import _parent_name, check_min_max_valid
from .utils import (
def _convert_equalization_ref(model: GraphModule):
    """ Reference function which applies changes needed for equalization, but
    does not quantize the nodes
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    weight_eq_obs_dict = update_obs_for_equalization(model, modules)
    convert_eq_obs(model, modules, weight_eq_obs_dict)
    return GraphModule(model, model.graph)