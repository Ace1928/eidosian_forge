import os
import tempfile
import unittest
import numpy as np
import numpy.testing as npt
import onnx
import onnx.helper
import onnx.model_container
import onnx.numpy_helper
import onnx.reference
class TestLargeOnnxReferenceEvaluator(unittest.TestCase):

    def common_check_reference_evaluator(self, container):
        X = np.arange(9).astype(np.float32).reshape((-1, 3))
        ref = onnx.reference.ReferenceEvaluator(container)
        got = ref.run(None, {'X': X})
        expected = np.array([[945000, 1015200, 1085400], [2905200, 3121200, 3337200], [4865400, 5227200, 5589000]], dtype=np.float32)
        npt.assert_allclose(expected, got[0])

    def test_large_onnx_no_large_initializer(self):
        model_proto = _linear_regression()
        large_model = onnx.model_container.make_large_model(model_proto.graph)
        self.common_check_reference_evaluator(large_model)
        with self.assertRaises(ValueError):
            large_model['#anymissingkey']
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, 'model.onnx')
            large_model.save(filename)
            copy = onnx.model_container.ModelContainer()
            copy.load(filename)
            self.common_check_reference_evaluator(copy)

    def test_large_one_weight_file(self):
        large_model = _large_linear_regression()
        self.common_check_reference_evaluator(large_model)
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, 'model.onnx')
            large_model.save(filename, True)
            copy = onnx.model_container.ModelContainer()
            copy.load(filename)
            loaded_model = onnx.load_model(filename, load_external_data=True)
            self.common_check_reference_evaluator(loaded_model)

    def test_large_multi_files(self):
        large_model = _large_linear_regression()
        self.common_check_reference_evaluator(large_model)
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, 'model.onnx')
            large_model.save(filename, False)
            copy = onnx.load_model(filename)
            self.common_check_reference_evaluator(copy)
            loaded_model = onnx.load_model(filename, load_external_data=True)
            self.common_check_reference_evaluator(loaded_model)