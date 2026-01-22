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
def _do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata):
    """Runs a step based on the given fetches and feeds.

    Args:
      handle: a handle for partial_run. None if this is just a call to run().
      target_list: A list of operations to be run, but not fetched.
      fetch_list: A list of tensors to be fetched.
      feed_dict: A dictionary that maps tensors to numpy ndarrays.
      options: A (pointer to a) [`RunOptions`] protocol buffer, or None
      run_metadata: A (pointer to a) [`RunMetadata`] protocol buffer, or None

    Returns:
      A list of numpy ndarrays, corresponding to the elements of
      `fetch_list`.  If the ith element of `fetch_list` contains the
      name of an operation, the first Tensor output of that operation
      will be returned for that element.

    Raises:
      tf.errors.OpError: Or one of its subclasses on error.
    """
    feeds = dict(((t.deref()._as_tf_output(), v) for t, v in feed_dict.items()))
    fetches = [t._as_tf_output() for t in fetch_list]
    targets = [op._c_op for op in target_list]

    def _run_fn(feed_dict, fetch_list, target_list, options, run_metadata):
        self._extend_graph()
        return self._call_tf_sessionrun(options, feed_dict, fetch_list, target_list, run_metadata)

    def _prun_fn(handle, feed_dict, fetch_list):
        if target_list:
            raise RuntimeError(f'partial_run() requires empty `target_list`. Received: target_list={target_list} (non-empty)')
        return self._call_tf_sessionprun(handle, feed_dict, fetch_list)
    if handle is None:
        return self._do_call(_run_fn, feeds, fetches, targets, options, run_metadata)
    else:
        return self._do_call(_prun_fn, handle, feeds, fetches)