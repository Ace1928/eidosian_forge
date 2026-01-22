import enum
import functools
from tensorflow.core.protobuf import data_service_pb2
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import compression_ops
from tensorflow.python.data.experimental.service import _pywrap_server_lib
from tensorflow.python.data.experimental.service import _pywrap_utils
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.ops.options import AutoShardPolicy
from tensorflow.python.data.ops.options import ExternalStatePolicy
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
def _parse_service(service):
    """Converts a tf.data service string into a (protocol, address) tuple.

  Args:
    service: A string in the format "protocol://address" or just "address". If
      the string is only an address, the default protocol will be used.

  Returns:
    The (protocol, address) tuple
  """
    if not isinstance(service, str):
        raise ValueError(f'`service` must be a string, but `service` was of type {type(service)}. service={service}')
    if not service:
        raise ValueError('`service` must not be empty')
    parts = service.split('://')
    if len(parts) == 2:
        protocol, address = parts
    elif len(parts) == 1:
        address = parts[0]
        protocol = _pywrap_utils.TF_DATA_DefaultProtocol()
    else:
        raise ValueError(f"Malformed `service` string has multiple '://': {service}.")
    return (protocol, address)