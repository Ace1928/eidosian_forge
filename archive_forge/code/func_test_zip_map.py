from __future__ import annotations
import itertools
import unittest
from typing import Any, Sequence
import numpy as np
import pytest
from parameterized import parameterized
import onnx.shape_inference
from onnx import (
from onnx.defs import (
from onnx.helper import (
from onnx.parser import parse_graph
@unittest.skipUnless(ONNX_ML, 'ONNX_ML required to test ai.onnx.ml operators')
def test_zip_map(self) -> None:
    params = (({'classlabels_int64s': [1, 2, 3]}, onnx.TensorProto.INT64), ({'classlabels_strings': ['a', 'b', 'c']}, onnx.TensorProto.STRING))
    for attrs, input_type in params:
        with self.subTest(attrs=attrs, input_type=input_type):
            self.zip_map_test_case(attrs, input_type)