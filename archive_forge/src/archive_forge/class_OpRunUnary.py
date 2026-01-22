from typing import Any, Dict
import numpy as np
from onnx.onnx_pb import NodeProto
from onnx.reference.op_run import OpRun, RuntimeTypeError
class OpRunUnary(OpRun):
    """Ancestor to all unary operators in this subfolder.

    Checks that input and output types are the same.
    """

    def run(self, x):
        """Calls method ``_run``, catches exceptions, displays a longer error message.

        Supports only unary operators.
        """
        self._log('-- begin %s.run(1 input)', self.__class__.__name__)
        try:
            res = self._run(x)
        except TypeError as e:
            raise TypeError(f'Issues with types {', '.join((str(type(_)) for _ in [x]))} (unary operator {self.__class__.__name__!r}).') from e
        self._log('-- done %s.run -> %d outputs', self.__class__.__name__, len(res))
        return self._check_and_fix_outputs(res)