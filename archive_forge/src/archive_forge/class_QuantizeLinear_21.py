from __future__ import annotations
from typing import ClassVar
import numpy as np
from onnx import TensorProto, subbyte
from onnx.helper import (
from onnx.reference.custom_element_types import (
from onnx.reference.op_run import OpRun
class QuantizeLinear_21(_CommonQuantizeLinear):

    def _run(self, *args, axis=None, saturate=None, block_size=None, output_dtype=None):
        return super()._run(*args, axis=axis, saturate=saturate, block_size=block_size, output_dtype=output_dtype)