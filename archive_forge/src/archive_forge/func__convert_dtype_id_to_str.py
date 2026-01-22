import collections
import os
import re
from packaging import version
from tensorflow.compiler.tf2tensorrt import _pywrap_py_utils
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import dtypes
def _convert_dtype_id_to_str(dtype):
    """Helper function to convert a dtype id to a corresponding string name."""
    if isinstance(dtype, int):
        return dtypes._TYPE_TO_STRING[dtype]
    else:
        return [dtypes._TYPE_TO_STRING[d] for d in dtype]