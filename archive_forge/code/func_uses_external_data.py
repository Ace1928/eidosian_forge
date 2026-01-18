import os
import re
import sys
import uuid
from itertools import chain
from typing import Callable, Iterable, Optional
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx.onnx_pb import AttributeProto, GraphProto, ModelProto, TensorProto
def uses_external_data(tensor: TensorProto) -> bool:
    """Returns true if the tensor stores data in an external location."""
    return tensor.HasField('data_location') and tensor.data_location == TensorProto.EXTERNAL