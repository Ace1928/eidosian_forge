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
def test_where_bfloat(self) -> None:
    graph = self._make_graph([('cond', TensorProto.BOOL, (10,)), ('x', TensorProto.BFLOAT16, (10,)), ('y', TensorProto.BFLOAT16, (10,))], [make_node('Where', ['cond', 'x', 'y'], ['out'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.BFLOAT16, (10,))])