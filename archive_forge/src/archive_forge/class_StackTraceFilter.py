import collections
import inspect
import threading
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.python.util import _tf_stack
class StackTraceFilter(StackTraceTransform):
    """Allows filtering traceback information by removing superfluous frames."""
    _stack_dict = _source_filter_stacks

    def __init__(self):
        self.internal_set = _tf_stack.PyBindFileSet()

    def update(self):
        self.internal_set.update_to(set(self.get_filtered_filenames()))

    def get_filtered_filenames(self):
        raise NotImplementedError('subclasses need to override this')