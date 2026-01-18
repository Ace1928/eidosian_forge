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
@parameterized.expand(all_versions_for('Transpose'))
def test_transpose_incorrect_repeated_perm(self, *_) -> None:
    graph = self._make_graph([('X', TensorProto.FLOAT, (2, 3, 4))], [make_node('Transpose', ['X'], ['Y'], perm=[1, 0, 1])], [])
    self.assertRaises(onnx.shape_inference.InferenceError, self._inferred, graph)