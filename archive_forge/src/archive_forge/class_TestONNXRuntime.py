import unittest
class TestONNXRuntime(unittest.TestCase):

    def test_with_ort_example(self) -> None:
        try:
            import onnxruntime
            del onnxruntime
        except ImportError:
            raise unittest.SkipTest('onnxruntime not installed') from None
        from numpy import float32, random
        from onnxruntime import InferenceSession
        from onnxruntime.datasets import get_example
        from onnx import checker, load, shape_inference, version_converter
        example1 = get_example('sigmoid.onnx')
        model = load(example1)
        checker.check_model(model)
        checker.check_model(model, full_check=True)
        inferred_model = shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=True)
        converted_model = version_converter.convert_version(inferred_model, 10)
        sess = InferenceSession(converted_model.SerializeToString(), providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        x = random.random((3, 4, 5))
        x = x.astype(float32)
        sess.run([output_name], {input_name: x})