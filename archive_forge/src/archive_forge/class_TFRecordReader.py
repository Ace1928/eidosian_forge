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
@tf_export(v1=['TFRecordReader'])
class TFRecordReader(ReaderBase):
    """A Reader that outputs the records from a TFRecords file.

  See ReaderBase for supported methods.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  """

    @deprecation.deprecated(None, 'Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.TFRecordDataset`.')
    def __init__(self, name=None, options=None):
        """Create a TFRecordReader.

    Args:
      name: A name for the operation (optional).
      options: A TFRecordOptions object (optional).
    """
        compression_type = python_io.TFRecordOptions.get_compression_type_string(options)
        rr = gen_io_ops.tf_record_reader_v2(name=name, compression_type=compression_type)
        super(TFRecordReader, self).__init__(rr)