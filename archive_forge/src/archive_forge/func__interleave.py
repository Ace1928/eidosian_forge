import os
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import parsing_ops
from tensorflow.python.util import compat
def _interleave(self, iterators, cycle_length):
    pending_iterators = iterators
    open_iterators = []
    num_open = 0
    for i in range(cycle_length):
        if pending_iterators:
            open_iterators.append(pending_iterators.pop(0))
            num_open += 1
    while num_open:
        for i in range(min(cycle_length, len(open_iterators))):
            if open_iterators[i] is None:
                continue
            try:
                yield next(open_iterators[i])
            except StopIteration:
                if pending_iterators:
                    open_iterators[i] = pending_iterators.pop(0)
                else:
                    open_iterators[i] = None
                    num_open -= 1