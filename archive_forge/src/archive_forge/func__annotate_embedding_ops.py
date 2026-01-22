from __future__ import annotations
import copy
from typing import List, Set
import torch
import torch.nn.functional as F
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from torch.ao.quantization.quantizer.quantizer import (
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
def _annotate_embedding_ops(self, graph: torch.fx.Graph) -> None:
    embedding_config: OperatorConfig = get_embedding_operators_config()
    for node in graph.nodes:
        if node.op == 'call_function' and node.target == torch.ops.aten.embedding.default:
            if embedding_config.config.weight is None:
                raise ValueError('Embedding config must have a valid weight quantization spec.')
            node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={node.args[0]: embedding_config.config.weight})