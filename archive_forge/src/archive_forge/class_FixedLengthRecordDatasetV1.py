import os
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import from_tensor_slices_op
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import convert
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['data.FixedLengthRecordDataset'])
class FixedLengthRecordDatasetV1(dataset_ops.DatasetV1Adapter):
    """A `Dataset` of fixed-length records from one or more binary files."""

    def __init__(self, filenames, record_bytes, header_bytes=None, footer_bytes=None, buffer_size=None, compression_type=None, num_parallel_reads=None, name=None):
        wrapped = FixedLengthRecordDatasetV2(filenames, record_bytes, header_bytes, footer_bytes, buffer_size, compression_type, num_parallel_reads, name=name)
        super(FixedLengthRecordDatasetV1, self).__init__(wrapped)
    __init__.__doc__ = FixedLengthRecordDatasetV2.__init__.__doc__

    @property
    def _filenames(self):
        return self._dataset._filenames

    @_filenames.setter
    def _filenames(self, value):
        self._dataset._filenames = value