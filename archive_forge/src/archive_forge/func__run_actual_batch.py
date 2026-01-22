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
def _run_actual_batch(self, outputs, label_key_provided=False):
    if label_key_provided:
        features, label = self.evaluate(outputs())
    else:
        features = self.evaluate(outputs())
        label = features['label']
    file_out = features['file']
    keywords_indices = features['keywords'].indices
    keywords_values = features['keywords'].values
    keywords_dense_shape = features['keywords'].dense_shape
    record = features['record']
    return [file_out, keywords_indices, keywords_values, keywords_dense_shape, record, label]