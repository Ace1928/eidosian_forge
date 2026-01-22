from typing import Any, Dict
import numpy as np
from onnx.onnx_pb import NodeProto
from onnx.reference.op_run import OpRun, RuntimeTypeError
class OpRunBinaryNumpy(OpRunBinaryNum):
    """*numpy_fct* is a binary numpy function which
    takes two matrices.
    """

    def __init__(self, numpy_fct: Any, onnx_node: NodeProto, run_params: Dict[str, Any]):
        OpRunBinaryNum.__init__(self, onnx_node, run_params)
        self.numpy_fct = numpy_fct

    def _run(self, a, b):
        res = (self.numpy_fct(a, b),)
        return self._check_and_fix_outputs(res)