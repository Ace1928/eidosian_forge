from typing import Any, List
import numpy as np
from onnx.reference.op_run import OpRun
class ConcatFromSequence(OpRun):

    def _run(self, seq, axis=None, new_axis=None):
        if seq is None:
            raise RuntimeError('A sequence cannot be null.')
        res = _concat_from_sequence(seq, axis, new_axis=new_axis)
        return (res,)