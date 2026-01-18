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
def test_tensor_dtype_to_storage_tensor_dtype_bfloat16(self) -> None:
    self.assertEqual(helper.tensor_dtype_to_storage_tensor_dtype(TensorProto.BFLOAT16), TensorProto.UINT16)