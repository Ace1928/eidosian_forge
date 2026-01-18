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
def test_sequence_length(self) -> None:
    graph = self._make_graph([('input1', TensorProto.FLOAT, (2, 3, 'x')), ('input2', TensorProto.FLOAT, (2, 3, 'x')), ('input3', TensorProto.FLOAT, (2, 3, 'x'))], [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']), make_node('SequenceLength', ['in_sequence'], ['len'])], [])
    self._assert_inferred(graph, [make_tensor_sequence_value_info('in_sequence', TensorProto.FLOAT, (2, 3, 'x')), make_tensor_value_info('len', TensorProto.INT64, ())])