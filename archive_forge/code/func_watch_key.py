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
@property
def watch_key(self):
    """Watch key identities a debug watch on a tensor.

    Returns:
      (`str`) A watch key, in the form of `tensor_name`:`debug_op`.
    """
    return _get_tensor_watch_key(self.node_name, self.output_slot, self.debug_op)