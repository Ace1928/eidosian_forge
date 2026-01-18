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
def test_stft_reals(self):
    graph = self._make_graph([], [make_node('Constant', [], ['signal'], value=make_tensor('signal', TensorProto.FLOAT, (2, 10, 1), (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3))), make_node('Constant', [], ['frame_step'], value=make_tensor('frame_step', TensorProto.INT64, (), (2,))), make_node('Constant', [], ['window'], value=make_tensor('window', TensorProto.INT64, (5,), (1, 2, 3, 4, 5))), make_node('STFT', ['signal', 'frame_step', 'window'], ['output'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('signal', TensorProto.FLOAT, (2, 10, 1)), make_tensor_value_info('frame_step', TensorProto.INT64, ()), make_tensor_value_info('window', TensorProto.INT64, (5,)), make_tensor_value_info('output', TensorProto.FLOAT, (2, 3, 5, 2))])
    graph = self._make_graph([], [make_node('Constant', [], ['signal'], value=make_tensor('signal', TensorProto.FLOAT, (2, 10, 1), (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3))), make_node('Constant', [], ['frame_step'], value=make_tensor('frame_step', TensorProto.INT64, (), (2,))), make_node('Constant', [], ['window'], value=make_tensor('window', TensorProto.INT64, (5,), (1, 2, 3, 4, 5))), make_node('Constant', [], ['frame_length'], value=make_tensor('frame_length', TensorProto.INT64, (), (5,))), make_node('STFT', ['signal', 'frame_step', 'window'], ['output'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('signal', TensorProto.FLOAT, (2, 10, 1)), make_tensor_value_info('frame_step', TensorProto.INT64, ()), make_tensor_value_info('window', TensorProto.INT64, (5,)), make_tensor_value_info('frame_length', TensorProto.INT64, ()), make_tensor_value_info('output', TensorProto.FLOAT, (2, 3, 5, 2))])
    graph = self._make_graph([], [make_node('Constant', [], ['signal'], value=make_tensor('signal', TensorProto.FLOAT, (2, 10, 1), (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3))), make_node('Constant', [], ['frame_step'], value=make_tensor('frame_step', TensorProto.INT64, (), (2,))), make_node('Constant', [], ['frame_length'], value=make_tensor('frame_length', TensorProto.INT64, (), (5,))), make_node('STFT', ['signal', 'frame_step', '', 'frame_length'], ['output'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('signal', TensorProto.FLOAT, (2, 10, 1)), make_tensor_value_info('frame_step', TensorProto.INT64, ()), make_tensor_value_info('frame_length', TensorProto.INT64, ()), make_tensor_value_info('output', TensorProto.FLOAT, (2, 3, 5, 2))])