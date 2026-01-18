from __future__ import annotations
from typing import ClassVar
import numpy as np
from onnx import TensorProto, subbyte
from onnx.helper import (
from onnx.reference.custom_element_types import (
from onnx.reference.op_run import OpRun
Reshape/Replicate scale/zero-point to be broadcastable to shape.

    Args:
        value: the array to be reshaped/replicated
        shape: the rarget shape
        axis: quantization axis, applicable for per-axis and blocked quantization
        block_size: size of quantization block, applicable only for blocked quantization

    Returns:
        value array after reshape/replicate according to quantization mode.
    