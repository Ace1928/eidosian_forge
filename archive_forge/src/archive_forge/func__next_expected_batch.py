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
def _next_expected_batch(self, file_indices, batch_size, num_epochs, cycle_length, drop_final_batch, use_parser_fn):

    def _next_record(file_indices):
        for j in file_indices:
            for i in range(self._num_records):
                yield (j, i)

    def _next_record_interleaved(file_indices, cycle_length):
        return self._interleave([_next_record([i]) for i in file_indices], cycle_length)
    record_batch = []
    batch_index = 0
    for _ in range(num_epochs):
        if cycle_length == 1:
            next_records = _next_record(file_indices)
        else:
            next_records = _next_record_interleaved(file_indices, cycle_length)
        for f, r in next_records:
            record = self._record(f, r)
            if use_parser_fn:
                record = record[1:]
            record_batch.append(record)
            batch_index += 1
            if len(record_batch) == batch_size:
                yield record_batch
                record_batch = []
                batch_index = 0
    if record_batch and (not drop_final_batch):
        yield record_batch