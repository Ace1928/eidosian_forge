import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, Set, Tuple
import numpy as np
import onnx
from onnx import ModelProto, ValueInfoProto, numpy_helper
from ..utils import logging, recurse_getattr
def _infer_output_shape(output: ValueInfoProto):
    """
    TODO: short documentation.
    """
    output_shape = []
    for dim in output.type.tensor_type.shape.dim:
        if getattr(dim, 'dim_param'):
            output_shape.append(getattr(dim, 'dim_param'))
        elif getattr(dim, 'dim_value'):
            output_shape.append(getattr(dim, 'dim_value'))
        else:
            raise ValueError('Cannot find `dim_param` nor `dim_value` in the output dimension info.')
    return output_shape