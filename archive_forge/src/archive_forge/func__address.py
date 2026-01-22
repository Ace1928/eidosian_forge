import collections
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.data.experimental.service import _pywrap_server_lib
from tensorflow.python.data.experimental.service import _pywrap_utils
from tensorflow.python.util.tf_export import tf_export
@property
def _address(self):
    """Returns the address of the server.

    The returned string will be in the form address:port, e.g. "localhost:1000".
    """
    return 'localhost:{0}'.format(self._server.bound_port())