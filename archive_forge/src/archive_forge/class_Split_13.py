from onnx.reference.op_run import OpRun
class Split_13(CommonSplit):

    def _run(self, mat, split=None, axis=None):
        return self.common_run(mat, split, axis=axis, num_outputs=None)