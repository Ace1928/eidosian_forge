from __future__ import annotations
from typing import ClassVar
import numpy as np
from onnx import TensorProto, subbyte
from onnx.helper import (
from onnx.reference.custom_element_types import (
from onnx.reference.op_run import OpRun
class QuantizeLinear_19(_CommonQuantizeLinear):

    def _run(self, x, y_scale, zero_point=None, axis=None, saturate=None):
        if len(y_scale.shape) > 1:
            raise ValueError('Input 2 must be a vector or a number.')
        return super()._run(x, y_scale, zero_point, axis=axis, saturate=saturate)