from typing import Any, Dict
import numpy as np
from onnx.onnx_pb import NodeProto
from onnx.reference.op_run import OpRun, RuntimeTypeError
class OpRunUnaryNum(OpRunUnary):
    """Ancestor to all unary and numerical operators in this subfolder.

    Checks that input and output types are the same.
    """

    def run(self, x):
        """Calls method ``OpRunUnary.run``.

        Catches exceptions, displays a longer error message.
        Checks that the result is not empty.
        """
        res = OpRunUnary.run(self, x)
        if len(res) == 0 or res[0] is None:
            return res
        if not isinstance(res[0], list) and res[0].dtype != x.dtype:
            raise RuntimeTypeError(f"Output type mismatch: input '{x.dtype}' != output '{res[0].dtype}' (operator {self.__class__.__name__!r}).")
        return self._check_and_fix_outputs(res)