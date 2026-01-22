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
def _verify_records(self, outputs, batch_size, file_index, num_epochs, interleave_cycle_length, drop_final_batch, use_parser_fn):
    if file_index is not None:
        if isinstance(file_index, list):
            file_indices = file_index
        else:
            file_indices = [file_index]
    else:
        file_indices = range(self._num_files)
    for expected_batch in self._next_expected_batch(file_indices, batch_size, num_epochs, interleave_cycle_length, drop_final_batch, use_parser_fn):
        actual_batch = self.evaluate(outputs())
        self.assertAllEqual(expected_batch, actual_batch)