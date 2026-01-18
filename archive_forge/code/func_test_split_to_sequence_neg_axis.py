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
def test_split_to_sequence_neg_axis(self) -> None:
    graph = self._make_graph([('input', TensorProto.FLOAT, (6, 4))], [make_node('SplitToSequence', ['input'], ['output_sequence'], axis=-2)], [])
    self._assert_inferred(graph, [make_tensor_sequence_value_info('output_sequence', TensorProto.FLOAT, (1, 4))])