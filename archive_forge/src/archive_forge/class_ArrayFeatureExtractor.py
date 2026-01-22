from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl
class ArrayFeatureExtractor(OpRunAiOnnxMl):

    def _run(self, data, indices):
        """Runtime for operator *ArrayFeatureExtractor*.

        Warning:
            ONNX specifications may be imprecise in some cases.
            When the input data is a vector (one dimension),
            the output has still two like a matrix with one row.
            The implementation follows what onnxruntime does in
            `array_feature_extractor.cc
            <https://github.com/microsoft/onnxruntime/blob/main/
            onnxruntime/core/providers/cpu/ml/array_feature_extractor.cc#L84>`_.
        """
        res = _array_feature_extrator(data, indices)
        return (res,)