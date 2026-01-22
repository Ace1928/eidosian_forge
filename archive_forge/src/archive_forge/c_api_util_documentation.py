import contextlib
from tensorflow.core.framework import api_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
Yields the managed C-API Object, guaranteeing aliveness.

    This is a context manager. Inside the context the C-API object is
    guaranteed to be alive.

    Raises:
      AlreadyGarbageCollectedError: if the object is already deleted.
    