import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def _test_window_function(self, window_function_name: str) -> None:
    size = helper.make_tensor('a', TensorProto.INT64, dims=[], vals=np.array([10]))
    self._test_op_upgrade(window_function_name, 17, [[]], [[10]], [TensorProto.INT64], initializer=[size])