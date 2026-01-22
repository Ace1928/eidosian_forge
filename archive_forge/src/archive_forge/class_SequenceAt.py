from onnx.reference.op_run import OpRun
class SequenceAt(OpRun):

    def _run(self, seq, index):
        return (seq[index],)