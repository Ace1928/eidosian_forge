from __future__ import annotations
import string
import unittest
from typing import Any, List, Sequence, cast
import onnx
from onnx import TensorProto, ValueInfoProto, helper, shape_inference, version_converter
def _test_model_conversion_fails(self, to_opset: int, model: str | onnx.ModelProto) -> None:
    if isinstance(model, str):
        model = onnx.parser.parse_model(model)
    onnx.checker.check_model(model)
    shape_inference.infer_shapes(model, strict_mode=True)
    with self.assertRaises(RuntimeError):
        version_converter.convert_version(model, to_opset)