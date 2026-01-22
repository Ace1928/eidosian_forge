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
def _callable_template_with_options_and_metadata(fetch_list, target_list, fetch_handler, options=None, run_metadata=None):
    """Template callable that accepts RunOptions and RunMetadata."""
    options_ptr = tf_session.TF_NewBufferFromString(compat.as_bytes(options.SerializeToString())) if options else None
    run_metadata_ptr = tf_session.TF_NewBuffer() if run_metadata else None
    try:
        results = self._call_tf_sessionrun(options_ptr, {}, fetch_list, target_list, run_metadata_ptr)
        if fetch_handler:
            results = fetch_handler.build_results(self, results)
        else:
            results = results[0] if results else None
        if run_metadata:
            proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)
            run_metadata.ParseFromString(compat.as_bytes(proto_data))
    finally:
        if run_metadata_ptr:
            tf_session.TF_DeleteBuffer(run_metadata_ptr)
        if options:
            tf_session.TF_DeleteBuffer(options_ptr)
    return results