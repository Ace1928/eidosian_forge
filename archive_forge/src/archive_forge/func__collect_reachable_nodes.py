from __future__ import annotations
import os
import onnx.checker
import onnx.helper
import onnx.shape_inference
from onnx import FunctionProto, ModelProto, NodeProto, TensorProto, ValueInfoProto
def _collect_reachable_nodes(self, input_names: list[str], output_names: list[str]) -> list[NodeProto]:
    reachable_nodes = []
    for name in output_names:
        self._dfs_search_reachable_nodes(name, input_names, reachable_nodes)
    nodes = [n for n in self.graph.node if n in reachable_nodes]
    return nodes