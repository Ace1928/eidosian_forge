import numpy as np
from tensorflow.core.framework import resource_handle_pb2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def to_numpy_array(self):
    """Convert a TensorHandle object to a feedable numpy value.

    Returns:
      A numpy array of a custom struct type that can be used as a feed value
      to run().
    """
    return encode_resource_handle(self._get_resource_handle())