from __future__ import annotations
import os
from typing import Sequence
import onnx
import onnx.onnx_cpp2py_export.shape_inference as C  # noqa: N812
from onnx import AttributeProto, FunctionProto, ModelProto, TypeProto
def to_type_proto(x) -> TypeProto:
    type_proto = onnx.TypeProto()
    type_proto.ParseFromString(x)
    return type_proto