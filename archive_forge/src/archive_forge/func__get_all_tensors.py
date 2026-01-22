import os
import re
import sys
import uuid
from itertools import chain
from typing import Callable, Iterable, Optional
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx.onnx_pb import AttributeProto, GraphProto, ModelProto, TensorProto
def _get_all_tensors(onnx_model_proto: ModelProto) -> Iterable[TensorProto]:
    """Scan an ONNX model for all tensors and return as an iterator."""
    return chain(_get_initializer_tensors(onnx_model_proto), _get_attribute_tensors(onnx_model_proto))