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
class _TFRecordDataset(dataset_ops.DatasetSource):
    """A `Dataset` comprising records from one or more TFRecord files."""

    def __init__(self, filenames, compression_type=None, buffer_size=None, name=None):
        """Creates a `TFRecordDataset`.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      buffer_size: (Optional.) A `tf.int64` scalar representing the number of
        bytes in the read buffer. 0 means no buffering.
      name: (Optional.) A name for the tf.data operation.
    """
        self._filenames = filenames
        self._compression_type = convert.optional_param_to_tensor('compression_type', compression_type, argument_default='', argument_dtype=dtypes.string)
        self._buffer_size = convert.optional_param_to_tensor('buffer_size', buffer_size, argument_default=_DEFAULT_READER_BUFFER_SIZE_BYTES)
        self._name = name
        variant_tensor = gen_dataset_ops.tf_record_dataset(self._filenames, self._compression_type, self._buffer_size, metadata=self._metadata.SerializeToString())
        super(_TFRecordDataset, self).__init__(variant_tensor)

    @property
    def element_spec(self):
        return tensor_spec.TensorSpec([], dtypes.string)