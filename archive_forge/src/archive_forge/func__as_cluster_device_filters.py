from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import device_filters_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import errors
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _as_cluster_device_filters(self):
    """Returns a serialized protobuf of cluster device filters."""
    if self._cluster_device_filters:
        return self._cluster_device_filters
    self._make_cluster_device_filters()
    return self._cluster_device_filters