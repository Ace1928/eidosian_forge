import torch
from torch._subclasses import FakeTensor
from torch.ao.quantization.fx.prepare import (
from torch.fx import (
from torch.fx.node import Argument
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
from typing import Dict, Tuple, Union, Any, Optional
from torch.ao.quantization.quantizer import (
from torch.ao.quantization import ObserverOrFakeQuantize
def _unwrap_shared_qspec(qspec: QuantizationSpecBase, edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpecBase], shared_with_map: Dict[EdgeOrNode, EdgeOrNode]) -> QuantizationSpecBase:
    """Unwraps qspec to get the final root qspec (non SharedQuantizationSpec)
    if qspec is SharedQuantizationSpec
       (1). tries to find the root edge or node for the node that the qspec points to
       (2). recursively find the root qspec based on the qspec for the root node
    """
    if isinstance(qspec, SharedQuantizationSpec):
        sharing_with = qspec.edge_or_node
        root = _find_root_edge_or_node(sharing_with, shared_with_map)
        qspec = edge_or_node_to_qspec[root]
        return _unwrap_shared_qspec(qspec, edge_or_node_to_qspec, shared_with_map)
    return qspec