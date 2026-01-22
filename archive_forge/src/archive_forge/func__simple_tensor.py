import io
import os
import pathlib
import tempfile
import unittest
import google.protobuf.message
import google.protobuf.text_format
import parameterized
import onnx
from onnx import serialization
def _simple_tensor() -> onnx.TensorProto:
    tensor = onnx.helper.make_tensor(name='test-tensor', data_type=onnx.TensorProto.FLOAT, dims=(2, 3, 4), vals=[x + 0.5 for x in range(24)])
    return tensor