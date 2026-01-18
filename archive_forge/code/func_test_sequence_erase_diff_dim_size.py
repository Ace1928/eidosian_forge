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
def test_sequence_erase_diff_dim_size(self) -> None:
    graph = self._make_graph([('input1', TensorProto.FLOAT, (2, 3, 'x')), ('input2', TensorProto.FLOAT, (2, 3, 'x')), ('input3', TensorProto.FLOAT, (2, 5, 'x')), ('ind', TensorProto.INT64, ())], [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']), make_node('SequenceErase', ['in_sequence', 'ind'], ['output_sequence'])], [])
    self._assert_inferred(graph, [make_tensor_sequence_value_info('in_sequence', TensorProto.FLOAT, (2, None, 'x')), make_tensor_sequence_value_info('output_sequence', TensorProto.FLOAT, (2, None, 'x'))])