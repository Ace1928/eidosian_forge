import os
import re
import sys
import uuid
from itertools import chain
from typing import Callable, Iterable, Optional
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx.onnx_pb import AttributeProto, GraphProto, ModelProto, TensorProto
def _recursive_attribute_processor(attribute: AttributeProto, func: Callable[[GraphProto], Iterable[TensorProto]]) -> Iterable[TensorProto]:
    """Create an iterator through processing ONNX model attributes with functor."""
    if attribute.type == AttributeProto.GRAPH:
        yield from func(attribute.g)
    if attribute.type == AttributeProto.GRAPHS:
        for graph in attribute.graphs:
            yield from func(graph)