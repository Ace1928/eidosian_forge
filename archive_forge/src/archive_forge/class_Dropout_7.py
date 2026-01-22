from typing import Optional, Tuple
import numpy as np
from numpy.random import RandomState  # type: ignore
from onnx.reference.op_run import OpRun
class Dropout_7(DropoutBase):

    def _run(self, X, ratio=None):
        return self._private_run(X, ratio)