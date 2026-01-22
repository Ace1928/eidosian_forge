import unittest
import automatic_conversion_test_base
import numpy as np
import parameterized
import onnx
from onnx import helper
def _test_op_downgrade(self, op: str, *args, **kwargs):
    self._test_op_conversion(op, *args, **kwargs, is_upgrade=False)