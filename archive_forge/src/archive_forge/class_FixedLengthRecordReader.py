from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops.gen_io_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['FixedLengthRecordReader'])
class FixedLengthRecordReader(ReaderBase):
    """A Reader that outputs fixed-length records from a file.

  See ReaderBase for supported methods.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  """

    @deprecation.deprecated(None, 'Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.FixedLengthRecordDataset`.')
    def __init__(self, record_bytes, header_bytes=None, footer_bytes=None, hop_bytes=None, name=None, encoding=None):
        """Create a FixedLengthRecordReader.

    Args:
      record_bytes: An int.
      header_bytes: An optional int. Defaults to 0.
      footer_bytes: An optional int. Defaults to 0.
      hop_bytes: An optional int. Defaults to 0.
      name: A name for the operation (optional).
      encoding: The type of encoding for the file. Defaults to none.
    """
        rr = gen_io_ops.fixed_length_record_reader_v2(record_bytes=record_bytes, header_bytes=header_bytes, footer_bytes=footer_bytes, hop_bytes=hop_bytes, encoding=encoding, name=name)
        super(FixedLengthRecordReader, self).__init__(rr)