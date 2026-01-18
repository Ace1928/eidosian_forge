import contextlib
from tensorflow.core.framework import api_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
@tf_contextlib.contextmanager
def tf_buffer(data=None):
    """Context manager that creates and deletes TF_Buffer.

  Example usage:
    with tf_buffer() as buf:
      # get serialized graph def into buf
      ...
      proto_data = c_api.TF_GetBuffer(buf)
      graph_def.ParseFromString(compat.as_bytes(proto_data))
    # buf has been deleted

    with tf_buffer(some_string) as buf:
      c_api.TF_SomeFunction(buf)
    # buf has been deleted

  Args:
    data: An optional `bytes`, `str`, or `unicode` object. If not None, the
      yielded buffer will contain this data.

  Yields:
    Created TF_Buffer
  """
    if data:
        buf = c_api.TF_NewBufferFromString(compat.as_bytes(data))
    else:
        buf = c_api.TF_NewBuffer()
    try:
        yield buf
    finally:
        c_api.TF_DeleteBuffer(buf)