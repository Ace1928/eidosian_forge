from __future__ import annotations
import os
import onnx.checker
import onnx.helper
import onnx.shape_inference
from onnx import FunctionProto, ModelProto, NodeProto, TensorProto, ValueInfoProto
def _collect_reachable_tensors(self, nodes: list[NodeProto]) -> tuple[list[TensorProto], list[ValueInfoProto]]:
    all_tensors_names: set[str] = set()
    for node in nodes:
        all_tensors_names.update(node.input)
        all_tensors_names.update(node.output)
    initializer = [self.wmap[t] for t in self.wmap if t in all_tensors_names]
    value_info = [self.vimap[t] for t in self.vimap if t in all_tensors_names]
    len_sparse_initializer = len(self.graph.sparse_initializer)
    if len_sparse_initializer != 0:
        raise ValueError(f'len_sparse_initializer is {len_sparse_initializer}, it must be 0.')
    len_quantization_annotation = len(self.graph.quantization_annotation)
    if len_quantization_annotation != 0:
        raise ValueError(f'len_quantization_annotation is {len_quantization_annotation}, it must be 0.')
    return (initializer, value_info)