import torch
from torch.ao.quantization.backend_config import BackendConfig
from torch.fx.graph import Node, Graph
from ..utils import _parent_name, NodePattern, Pattern
from ..fuser_method_mappings import get_fuser_method_new
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Union
from .custom_config import FuseCustomConfig
from .match_utils import MatchAllNode
from torch.nn.utils.parametrize import type_before_parametrizations
 Given a node pattern, extract the corresponding modules
            e.g. input: (relu_node, (bn_node, conv_node))
                 output: (relu_module, (bn_module, conv_module))
            