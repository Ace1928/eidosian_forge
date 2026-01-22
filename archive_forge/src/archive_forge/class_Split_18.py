from onnx.reference.op_run import OpRun
class Split_18(CommonSplit):

    def _run(self, mat, split=None, axis=None, num_outputs=None):
        return self.common_run(mat, split, axis=axis, num_outputs=num_outputs)