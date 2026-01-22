import os
import re
import sys
import uuid
from itertools import chain
from typing import Callable, Iterable, Optional
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx.onnx_pb import AttributeProto, GraphProto, ModelProto, TensorProto
def _get_initializer_tensors_from_graph(onnx_model_proto_graph: GraphProto) -> Iterable[TensorProto]:
    """Create an iterator of initializer tensors from ONNX model graph."""
    yield from onnx_model_proto_graph.initializer
    for node in onnx_model_proto_graph.node:
        for attribute in node.attribute:
            yield from _recursive_attribute_processor(attribute, _get_initializer_tensors_from_graph)