import os
import tempfile
import unittest
import onnx
def test_onnx_save_load_model_uses_the_custom_serializer(self) -> None:
    model = onnx.parser.parse_model(_TEST_MODEL)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'model.onnx')
        onnx.save_model(model, model_path, format='onnxtext')
        with open(model_path, encoding='utf-8') as f:
            content = f.read()
            self.assertEqual(content, onnx.printer.to_text(model))
        loaded_model = onnx.load_model(model_path, format='onnxtext')
        self.assertEqual(model.SerializeToString(deterministic=True), loaded_model.SerializeToString(deterministic=True))