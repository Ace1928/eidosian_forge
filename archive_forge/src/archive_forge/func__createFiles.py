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
def _createFiles(self):
    filenames = []
    for i in range(self._num_files):
        fn = os.path.join(self.get_temp_dir(), 'tf_record.%d.txt' % i)
        filenames.append(fn)
        writer = python_io.TFRecordWriter(fn)
        for j in range(self._num_records):
            writer.write(self._record(i, j))
        writer.close()
    return filenames