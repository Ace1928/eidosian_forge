import collections
import functools
import re
import threading
import warnings
import numpy as np
import wrapt
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import pywrap_tf_session as tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import device
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor
from tensorflow.python.ops import session_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.experimental import mixed_precision_global_state
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class _Callable(object):
    """Experimental wrapper for the C++ `Session::MakeCallable()` API."""

    def __init__(self, session, callable_options):
        self._session = session
        self._handle = None
        options_ptr = tf_session.TF_NewBufferFromString(compat.as_bytes(callable_options.SerializeToString()))
        try:
            self._handle = tf_session.TF_SessionMakeCallable(session._session, options_ptr)
        finally:
            tf_session.TF_DeleteBuffer(options_ptr)

    def __call__(self, *args, **kwargs):
        run_metadata = kwargs.get('run_metadata', None)
        try:
            run_metadata_ptr = tf_session.TF_NewBuffer() if run_metadata else None
            ret = tf_session.TF_SessionRunCallable(self._session._session, self._handle, args, run_metadata_ptr)
            if run_metadata:
                proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)
                run_metadata.ParseFromString(compat.as_bytes(proto_data))
        finally:
            if run_metadata_ptr:
                tf_session.TF_DeleteBuffer(run_metadata_ptr)
        return ret

    def __del__(self):
        if self._handle is not None and self._session._session is not None and (not self._session._closed):
            tf_session.TF_SessionReleaseCallable(self._session._session, self._handle)