import collections
import copy
import gc
from tensorflow.python.framework import _python_memory_checker_helper
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
def _create_python_object_snapshot():
    gc.collect()
    all_objects = gc.get_objects()
    result = collections.defaultdict(set)
    for obj in all_objects:
        result[_get_typename(obj)].add(id(obj))
    return result