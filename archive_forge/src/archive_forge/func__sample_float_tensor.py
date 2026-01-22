import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
@property
def _sample_float_tensor(self) -> TensorProto:
    np_array = np.random.randn(2, 3).astype(np.float32)
    return helper.make_tensor(name='test', data_type=TensorProto.FLOAT, dims=(2, 3), vals=np_array.reshape(6).tolist())