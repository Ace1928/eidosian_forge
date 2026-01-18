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
def test_reshape_inference(self) -> None:
    assert _run_case(RESHAPE_SCHEMA, ['x', 't'], ['y'], _to_tensor_types({'x': (TensorProto.FLOAT, (5, 4)), 't': (TensorProto.INT64, (3,))}), {'t': np.array([2, 2, 5], dtype=np.int64)}) == _to_tensor_types({'y': (TensorProto.FLOAT, (2, 2, 5))})