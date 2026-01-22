import os
import tempfile
import unittest
import numpy as np
import onnx
import onnx.external_data_helper as ext_data
import onnx.helper
import onnx.model_container
import onnx.numpy_helper
class TestLargeOnnx(unittest.TestCase):

    def test_large_onnx_no_large_initializer(self):
        model_proto = _linear_regression()
        assert isinstance(model_proto, onnx.ModelProto)
        large_model = onnx.model_container.make_large_model(model_proto.graph)
        assert isinstance(large_model, onnx.model_container.ModelContainer)
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, 'model.onnx')
            large_model.save(filename)
            copy = onnx.model_container.ModelContainer()
            with self.assertRaises(RuntimeError):
                assert copy.model_proto
            copy.load(filename)
            assert copy.model_proto is not None
            onnx.checker.check_model(copy.model_proto)

    def test_large_one_weight_file(self):
        large_model = _large_linear_regression()
        assert isinstance(large_model, onnx.model_container.ModelContainer)
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, 'model.onnx')
            saved_proto = large_model.save(filename, True)
            assert isinstance(saved_proto, onnx.ModelProto)
            copy = onnx.model_container.ModelContainer()
            copy.load(filename)
            copy.check_model()
            loaded_model = onnx.load_model(filename, load_external_data=True)
            onnx.checker.check_model(loaded_model)

    def test_large_multi_files(self):
        large_model = _large_linear_regression()
        assert isinstance(large_model, onnx.model_container.ModelContainer)
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, 'model.onnx')
            saved_proto = large_model.save(filename, False)
            assert isinstance(saved_proto, onnx.ModelProto)
            copy = onnx.load_model(filename)
            onnx.checker.check_model(copy)
            for tensor in ext_data._get_all_tensors(copy):
                if ext_data.uses_external_data(tensor):
                    tested = 0
                    for ext in tensor.external_data:
                        if ext.key == 'location':
                            assert os.path.exists(ext.value)
                            tested += 1
                    self.assertEqual(tested, 1)
            loaded_model = onnx.load_model(filename, load_external_data=True)
            onnx.checker.check_model(loaded_model)