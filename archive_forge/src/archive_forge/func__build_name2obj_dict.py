from __future__ import annotations
import os
import onnx.checker
import onnx.helper
import onnx.shape_inference
from onnx import FunctionProto, ModelProto, NodeProto, TensorProto, ValueInfoProto
@staticmethod
def _build_name2obj_dict(objs):
    return {obj.name: obj for obj in objs}