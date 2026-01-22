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
def _rnn_layout(self, seqlen: int, batchsize: int, inpsize: int, hiddensize: int, direction: str='forward') -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (batchsize, seqlen, inpsize)), ('w', TensorProto.FLOAT, (1, hiddensize, inpsize)), ('r', TensorProto.FLOAT, (1, hiddensize, hiddensize))], [make_node('RNN', ['x', 'w', 'r'], ['all', 'last'], hidden_size=hiddensize, layout=1, direction=direction)], [])
    if direction == 'bidirectional':
        num_directions = 2
    else:
        num_directions = 1
    self._assert_inferred(graph, [make_tensor_value_info('all', TensorProto.FLOAT, (batchsize, seqlen, num_directions, hiddensize)), make_tensor_value_info('last', TensorProto.FLOAT, (batchsize, num_directions, hiddensize))])