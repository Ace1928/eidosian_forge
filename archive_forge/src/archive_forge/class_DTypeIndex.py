import collections
import os
import re
from packaging import version
from tensorflow.compiler.tf2tensorrt import _pywrap_py_utils
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import dtypes
class DTypeIndex(dict):
    """Helper class to create an index of dtypes with incremental values."""

    def get_dtype_index(self, dtype):
        if dtype not in self:
            self[dtype] = len(self) + 1
        return self[dtype]