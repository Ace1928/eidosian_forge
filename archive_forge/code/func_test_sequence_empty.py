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
def test_sequence_empty(self) -> None:
    graph = self._make_graph([], [make_node('SequenceEmpty', [], ['output'])], [])
    self._assert_inferred(graph, [make_tensor_sequence_value_info('output', TensorProto.FLOAT, None)])