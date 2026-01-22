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
def _MediumAppendFloat16ArrayToTensorProto(tensor_proto, proto_values):
    fast_tensor_util.AppendFloat16ArrayToTensorProto(tensor_proto, np.asarray(proto_values, dtype=np.float16).view(np.uint16))