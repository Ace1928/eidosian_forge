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
def _find_partition_graph(self, partition_graphs, device_name):
    if partition_graphs is None:
        return None
    else:
        for graph_def in partition_graphs:
            for node_def in graph_def.node:
                if node_def.device == device_name:
                    return graph_def
        return None