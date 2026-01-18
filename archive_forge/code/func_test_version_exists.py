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
def test_version_exists(self) -> None:
    model = onnx.ModelProto()
    self.assertFalse(model.HasField('ir_version'))
    model.ir_version = onnx.IR_VERSION
    model_string = model.SerializeToString()
    model.ParseFromString(model_string)
    self.assertTrue(model.HasField('ir_version'))
    self.assertEqual(model.ir_version, onnx.IR_VERSION)