import json
import os
import sys
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import _pywrap_kernel_registry
Computes a header for use with tensorflow SELECTIVE_REGISTRATION.

  Args:
    graphs: a list of paths to GraphDef files to include.
    proto_fileformat: optional format of proto file, either 'textproto',
      'rawproto' (default) or ops_list. The ops_list is the file contain the
      list of ops in JSON format, Ex: "[["Transpose", "TransposeCpuOp"]]".
    default_ops: optional comma-separated string of operator:kernel pairs to
      always include implementation for. Pass 'all' to have all operators and
      kernels included. Default: 'NoOp:NoOp,_Recv:RecvOp,_Send:SendOp'.

  Returns:
    the string of the header that should be written as ops_to_register.h.
  