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
def test_save_model_uses_format_when_it_is_specified(self) -> None:
    proto = _simple_model()
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'model.textproto')
        onnx.save_model(proto, model_path, format='protobuf')
        loaded_proto = onnx.load_model(model_path, format='protobuf')
        self.assertEqual(proto, loaded_proto)
        with self.assertRaises(google.protobuf.text_format.ParseError):
            onnx.load_model(model_path)