import logging
from collections import defaultdict
from typing import Tuple, Dict, Optional, List
import torch
from torch._export import export
from torch._export.pass_base import _ExportPassBase
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue
from torch._subclasses import FakeTensor
from torch.fx.node import Target, Argument
from torch.library import Library
from torch.utils._pytree import tree_unflatten
import torch._export.exported_program as ep
import re
def _parse_upgraders(self, op_upgraders: Optional[Dict[str, Tuple[str, str]]]=None) -> List[Tuple[str, str]]:
    """Reorder op_upgraders by version number, return an ordered list of tuples, containing old op schema as well
        as the upgrader function string literal."""
    op_namespace = 'aten'
    if not op_upgraders or op_namespace not in self.model_opset_version or op_namespace not in self.compiler_opset_version:
        return []
    model_ver = self.model_opset_version[op_namespace]
    curr_ver = self.compiler_opset_version[op_namespace]
    versioned_upgraders: Dict[int, Tuple[str, str]] = {get_target_version(name): v for name, v in op_upgraders.items()}
    target_upgraders: List[Tuple[str, str]] = []
    for ver in range(model_ver + 1, curr_ver + 1):
        if ver in versioned_upgraders:
            target_upgraders.append(versioned_upgraders[ver])
        else:
            log.warning('Missing an upgrader to upgrade to version {ver}.', extra={'ver': ver})
    return target_upgraders