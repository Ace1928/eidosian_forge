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
def _get_fusion_pattern_to_fuse_handler_cls(backend_config: BackendConfig) -> Dict[Pattern, Callable]:
    fusion_pattern_to_fuse_handlers: Dict[Pattern, Callable] = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        if config.fuser_method is not None:
            fusion_pattern_to_fuse_handlers[pattern] = DefaultFuseHandler
    return fusion_pattern_to_fuse_handlers