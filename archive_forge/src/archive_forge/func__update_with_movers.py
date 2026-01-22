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
def _update_with_movers(self, feed_dict, feed_map):
    handle_movers = []
    for feed_name, val in feed_map.items():
        mover = session_ops._get_handle_mover(self.graph, *val)
        if mover:
            handle_movers.append((feed_name, val[1], mover))
    if not handle_movers:
        return []
    else:
        feeds = {}
        fetches = []
        for _, handle, mover in handle_movers:
            feeds[mover[0]] = handle
            fetches.append(mover[1])
        handles = self.run(fetches, feed_dict=feeds)
        for handle_mover, handle in zip(handle_movers, handles):
            np_val = np.array(handle.handle, dtype=np.object_)
            feed_name = handle_mover[0]
            feed_tensor = feed_map[feed_name][0]
            feed_dict[feed_tensor.ref()] = np_val
        return handles