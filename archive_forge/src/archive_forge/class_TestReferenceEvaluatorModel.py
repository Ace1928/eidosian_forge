import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import onnx.reference as orf
class TestReferenceEvaluatorModel(unittest.TestCase):

    def test_loop_fft(self):
        model = create_model()
        session = orf.ReferenceEvaluator(model)
        session.run(None, {'A': -np.arange(10).astype(np.float32)})