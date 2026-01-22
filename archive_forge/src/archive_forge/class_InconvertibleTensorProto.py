import collections
import glob
import json
import os
import platform
import re
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
class InconvertibleTensorProto:
    """Represents a TensorProto that cannot be converted to np.ndarray."""

    def __init__(self, tensor_proto, initialized=True):
        """Constructor.

    Args:
      tensor_proto: the `TensorProto` object that cannot be represented as a
        `np.ndarray` object.
      initialized: (`bool`) whether the Tensor is initialized.
    """
        self._tensor_proto = tensor_proto
        self._initialized = initialized

    def __str__(self):
        output = '' if self._initialized else 'Uninitialized tensor:\n'
        output += str(self._tensor_proto)
        return output

    @property
    def initialized(self):
        return self._initialized