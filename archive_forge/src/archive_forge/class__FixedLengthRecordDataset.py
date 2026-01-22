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
class _FixedLengthRecordDataset(dataset_ops.DatasetSource):
    """A `Dataset` of fixed-length records from one or more binary files."""

    def __init__(self, filenames, record_bytes, header_bytes=None, footer_bytes=None, buffer_size=None, compression_type=None, name=None):
        """Creates a `FixedLengthRecordDataset`.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      record_bytes: A `tf.int64` scalar representing the number of bytes in each
        record.
      header_bytes: (Optional.) A `tf.int64` scalar representing the number of
        bytes to skip at the start of a file.
      footer_bytes: (Optional.) A `tf.int64` scalar representing the number of
        bytes to ignore at the end of a file.
      buffer_size: (Optional.) A `tf.int64` scalar representing the number of
        bytes to buffer when reading.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      name: (Optional.) A name for the tf.data operation.
    """
        self._filenames = filenames
        self._record_bytes = ops.convert_to_tensor(record_bytes, dtype=dtypes.int64, name='record_bytes')
        self._header_bytes = convert.optional_param_to_tensor('header_bytes', header_bytes)
        self._footer_bytes = convert.optional_param_to_tensor('footer_bytes', footer_bytes)
        self._buffer_size = convert.optional_param_to_tensor('buffer_size', buffer_size, _DEFAULT_READER_BUFFER_SIZE_BYTES)
        self._compression_type = convert.optional_param_to_tensor('compression_type', compression_type, argument_default='', argument_dtype=dtypes.string)
        self._name = name
        variant_tensor = gen_dataset_ops.fixed_length_record_dataset_v2(self._filenames, self._header_bytes, self._record_bytes, self._footer_bytes, self._buffer_size, self._compression_type, metadata=self._metadata.SerializeToString())
        super(_FixedLengthRecordDataset, self).__init__(variant_tensor)

    @property
    def element_spec(self):
        return tensor_spec.TensorSpec([], dtypes.string)