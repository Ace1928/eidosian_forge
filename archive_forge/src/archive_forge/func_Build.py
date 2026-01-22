import collections
import inspect
import threading
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.python.util import _tf_stack
def Build(self):
    debug_info_str = super().Build()
    debug_info = graph_debug_info_pb2.GraphDebugInfo()
    debug_info.ParseFromString(debug_info_str)
    return debug_info