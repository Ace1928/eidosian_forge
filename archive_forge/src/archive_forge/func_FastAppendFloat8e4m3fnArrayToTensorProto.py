import typing
from typing import Protocol
import numpy as np
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import tensor_shape
from tensorflow.python.types import core
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def FastAppendFloat8e4m3fnArrayToTensorProto(tensor_proto, proto_values):
    fast_tensor_util.AppendFloat8ArrayToTensorProto(tensor_proto, np.asarray(proto_values, dtype=dtypes.float8_e4m3fn.as_numpy_dtype).view(np.uint8))