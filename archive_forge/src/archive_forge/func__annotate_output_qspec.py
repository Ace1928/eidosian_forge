from typing import List
from torch.ao.quantization.pt2e.utils import _is_sym_size_node
from torch.ao.quantization.quantizer.quantizer import QuantizationAnnotation
from torch.fx import Node
def _annotate_output_qspec(node: Node, qspec):
    quantization_annotation = node.meta.get('quantization_annotation', QuantizationAnnotation())
    quantization_annotation.output_qspec = qspec
    node.meta['quantization_annotation'] = quantization_annotation