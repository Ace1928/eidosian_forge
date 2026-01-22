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
def _get_edge_or_node_to_qspec(model: torch.fx.GraphModule) -> Dict[EdgeOrNode, QuantizationSpecBase]:
    """Get a map from EdgeOrNode to quantization spec based on annotations on the nodes
    """
    edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpecBase] = {}
    for n in model.graph.nodes:
        if hasattr(n, 'meta') and 'quantization_annotation' in n.meta:
            qa = n.meta['quantization_annotation']
            for input_to_n, qspec in qa.input_qspec_map.items():
                input_edge = (input_to_n, n)
                edge_or_node_to_qspec[input_edge] = qspec
            if qa.output_qspec is not None:
                output_node = n
                qspec = qa.output_qspec
                edge_or_node_to_qspec[output_node] = qspec
    return edge_or_node_to_qspec