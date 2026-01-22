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
def _get_merge_node_names(self, device_name):
    """Lazily get a list of Merge nodes on a given device."""
    if device_name not in self._device_names:
        raise ValueError('Invalid device name: %s' % device_name)
    if not hasattr(self, '_merge_node_names'):
        self._merge_node_names = {}
    if device_name not in self._merge_node_names:
        debug_graph = self._debug_graphs[device_name]
        self._merge_node_names[device_name] = [node for node in debug_graph.node_op_types if debug_graph.node_op_types[node] == 'Merge']
    return self._merge_node_names[device_name]