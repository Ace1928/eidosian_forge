import collections
import inspect
import threading
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.python.util import _tf_stack
class SentinelFilter(StackTraceFilter):

    def get_filtered_filenames(self):
        return EMPTY_SET