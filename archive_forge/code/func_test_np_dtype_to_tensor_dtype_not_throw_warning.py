import itertools
import random
import struct
import unittest
from typing import Any, List, Tuple
import numpy as np
import parameterized
import pytest
import version_utils
from onnx import (
from onnx.reference.op_run import to_array_extended
@pytest.mark.filterwarnings('error::DeprecationWarning')
def test_np_dtype_to_tensor_dtype_not_throw_warning(self) -> None:
    _ = helper.np_dtype_to_tensor_dtype(np.dtype('float32'))