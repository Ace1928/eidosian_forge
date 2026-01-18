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
def test_scan_opset9_axes(self) -> None:
    axis_0_len = 'axis0'
    seq_len = 'sequence'
    input_size = 2
    loop_state_size = 3
    input_value_infos = [make_tensor_value_info('loop_state_in', TensorProto.UNDEFINED, None), make_tensor_value_info('input', TensorProto.UNDEFINED, None)]
    output_value_infos = [make_tensor_value_info('loop_state_out', TensorProto.UNDEFINED, None), make_tensor_value_info('output', TensorProto.UNDEFINED, None)]
    subgraph = helper.make_graph([make_node('Identity', ['loop_state_in'], ['loop_state_out']), make_node('Identity', ['input'], ['output'])], 'subgraph', input_value_infos, output_value_infos)
    graph = self._make_graph([('loop_state_orig', TensorProto.FLOAT, (loop_state_size,)), ('scan_input', TensorProto.FLOAT, (axis_0_len, seq_len, input_size))], [make_node('Scan', ['loop_state_orig', 'scan_input'], ['loop_state_final', 'scan_output'], num_scan_inputs=1, body=subgraph, scan_input_axes=[1])], [])
    self._assert_inferred(graph, [make_tensor_value_info('loop_state_final', TensorProto.FLOAT, (loop_state_size,)), make_tensor_value_info('scan_output', TensorProto.FLOAT, (seq_len, axis_0_len, input_size))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 9)])