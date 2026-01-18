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
def test_topk_missing_k_value_output_rank_check(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (3, 4, 5, 10)), ('k', TensorProto.INT64, (1,))], [make_node('TopK', ['x', 'k'], ['y', 'z'], axis=2)], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (None, None, None, None)), make_tensor_value_info('z', TensorProto.INT64, (None, None, None, None))])