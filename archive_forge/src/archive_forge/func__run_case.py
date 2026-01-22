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
def _run_case(schema: OpSchema, input_names: List[str], output_names: List[str], input_types: Dict[str, TypeProto], input_data: Optional[Dict[str, np.ndarray]]=None) -> Dict[str, TypeProto]:
    if input_data is None:
        input_data = {}
    return infer_node_outputs(schema, make_node(schema.name, input_names, output_names, domain=schema.domain), input_types, {key: from_array(arr) for key, arr in input_data.items()})