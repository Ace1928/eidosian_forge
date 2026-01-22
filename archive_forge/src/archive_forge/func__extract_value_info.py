import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import onnx
from onnx.backend.test.case.test_case import TestCase
from onnx.backend.test.case.utils import import_recursive
from onnx.onnx_pb import (
def _extract_value_info(input: Union[List[Any], np.ndarray, None], name: str, type_proto: Optional[TypeProto]=None) -> onnx.ValueInfoProto:
    if type_proto is None:
        if input is None:
            raise NotImplementedError('_extract_value_info: both input and type_proto arguments cannot be None.')
        elif isinstance(input, list):
            elem_type = onnx.helper.np_dtype_to_tensor_dtype(input[0].dtype)
            shape = None
            tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type, shape)
            type_proto = onnx.helper.make_sequence_type_proto(tensor_type_proto)
        elif isinstance(input, TensorProto):
            elem_type = input.data_type
            shape = tuple(input.dims)
            type_proto = onnx.helper.make_tensor_type_proto(elem_type, shape)
        else:
            elem_type = onnx.helper.np_dtype_to_tensor_dtype(input.dtype)
            shape = input.shape
            type_proto = onnx.helper.make_tensor_type_proto(elem_type, shape)
    return onnx.helper.make_value_info(name, type_proto)