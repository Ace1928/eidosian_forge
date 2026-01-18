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
def test_save_and_load_model_when_input_has_read_function(self) -> None:
    proto = _simple_model()
    proto_string = serialization.registry.get('protobuf').serialize_proto(proto)
    f = io.BytesIO()
    onnx.save_model(proto_string, f, format=self.format)
    loaded_proto = onnx.load_model(io.BytesIO(f.getvalue()), format=self.format)
    self.assertEqual(proto, loaded_proto)