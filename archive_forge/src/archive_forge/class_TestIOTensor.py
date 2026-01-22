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
@parameterized.parameterized_class([{'format': 'protobuf'}, {'format': 'textproto'}, {'format': 'json'}])
class TestIOTensor(unittest.TestCase):
    """Test loading and saving of TensorProto."""
    format: str

    def test_load_tensor_when_input_is_bytes(self) -> None:
        proto = _simple_tensor()
        proto_string = serialization.registry.get(self.format).serialize_proto(proto)
        loaded_proto = onnx.load_tensor_from_string(proto_string, format=self.format)
        self.assertEqual(proto, loaded_proto)

    def test_save_and_load_tensor_when_input_has_read_function(self) -> None:
        proto = _simple_tensor()
        f = io.BytesIO()
        onnx.save_tensor(proto, f, format=self.format)
        loaded_proto = onnx.load_tensor(io.BytesIO(f.getvalue()), format=self.format)
        self.assertEqual(proto, loaded_proto)

    def test_save_and_load_tensor_when_input_is_file_name(self) -> None:
        proto = _simple_tensor()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.onnx')
            onnx.save_tensor(proto, model_path, format=self.format)
            loaded_proto = onnx.load_tensor(model_path, format=self.format)
            self.assertEqual(proto, loaded_proto)

    def test_save_and_load_tensor_when_input_is_pathlike(self) -> None:
        proto = _simple_tensor()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = pathlib.Path(temp_dir, 'model.onnx')
            onnx.save_tensor(proto, model_path, format=self.format)
            loaded_proto = onnx.load_tensor(model_path, format=self.format)
            self.assertEqual(proto, loaded_proto)