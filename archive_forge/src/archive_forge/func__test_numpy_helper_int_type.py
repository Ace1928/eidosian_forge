import unittest
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import helper, numpy_helper
def _test_numpy_helper_int_type(self, dtype: np.number) -> None:
    a = np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max, dtype=dtype, size=(13, 37))
    tensor_def = numpy_helper.from_array(a, 'test')
    self.assertEqual(tensor_def.name, 'test')
    a_recover = numpy_helper.to_array(tensor_def)
    np.testing.assert_equal(a, a_recover)