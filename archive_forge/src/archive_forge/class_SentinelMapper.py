import collections
import inspect
import threading
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.python.util import _tf_stack
class SentinelMapper(StackTraceMapper):

    def get_effective_source_map(self):
        return EMPTY_DICT