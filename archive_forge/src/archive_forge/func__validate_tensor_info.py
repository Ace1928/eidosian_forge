import functools
import os
from google.protobuf.any_pb2 import Any
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _validate_tensor_info(self, tensor_info):
    """Validates the `TensorInfo` proto.

    Checks if the `encoding` (`name` or `coo_sparse` or `type_spec`) and
    `dtype` fields exist and are non-empty.

    Args:
      tensor_info: `TensorInfo` protocol buffer to validate.

    Raises:
      AssertionError: If the `encoding` or `dtype` fields of the supplied
          `TensorInfo` proto are not populated.
    """
    if tensor_info is None:
        raise AssertionError('All TensorInfo protos used in the SignatureDefs must have the name and dtype fields set.')
    if tensor_info.WhichOneof('encoding') is None:
        raise AssertionError(f"Invalid `tensor_info`: {tensor_info}. All TensorInfo protos used in the SignatureDefs must have one of the 'encoding' fields (e.g., name or coo_sparse) set.")
    if tensor_info.WhichOneof('encoding') == 'composite_tensor':
        for component in tensor_info.composite_tensor.components:
            self._validate_tensor_info(component)
    elif tensor_info.dtype == types_pb2.DT_INVALID:
        raise AssertionError(f'Invalid `tensor_info`: {tensor_info}. All TensorInfo protos used in the SignatureDefs must have the dtype field set.')