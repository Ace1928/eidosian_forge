import unittest
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
from onnx import TensorProto, TypeProto
from onnx.checker import ValidationError
from onnx.defs import OpSchema, get_all_schemas_with_history, get_schema
from onnx.helper import (
from onnx.numpy_helper import from_array
from onnx.shape_inference import InferenceError, infer_node_outputs
def test_scan_inference_with_subgraph(self) -> None:
    seq_len = 'sequence'
    input_size = 2
    loop_state_size = 3
    input_value_infos = [make_tensor_value_info('loop_state_in', TensorProto.UNDEFINED, None), make_tensor_value_info('input', TensorProto.UNDEFINED, None), make_tensor_value_info('outer', TensorProto.UNDEFINED, None)]
    output_value_infos = [make_tensor_value_info('loop_state_out', TensorProto.UNDEFINED, None), make_tensor_value_info('output', TensorProto.FLOAT, (seq_len, input_size))]
    subgraph = make_graph([make_node('Identity', ['loop_state_in'], ['loop_state_out']), make_node('Add', ['input', 'outer'], ['output'])], 'subgraph', input_value_infos, output_value_infos)
    assert infer_node_outputs(get_schema('Scan', 9), make_node('Scan', ['loop_state_orig', 'scan_input', 'scan_outer'], ['loop_state_final', 'scan_output'], num_scan_inputs=1, body=subgraph), _to_tensor_types({'loop_state_orig': (TensorProto.FLOAT, (loop_state_size,)), 'scan_input': (TensorProto.FLOAT, (seq_len, input_size)), 'scan_outer': (TensorProto.FLOAT, (input_size,))}), opset_imports=[make_opsetid('', 9)], ir_version=4) == _to_tensor_types({'loop_state_final': (TensorProto.FLOAT, (loop_state_size,)), 'scan_output': (TensorProto.FLOAT, (seq_len, input_size))})