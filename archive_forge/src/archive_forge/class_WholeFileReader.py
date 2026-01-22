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
@tf_export(v1=['WholeFileReader'])
class WholeFileReader(ReaderBase):
    """A Reader that outputs the entire contents of a file as a value.

  To use, enqueue filenames in a Queue.  The output of Read will
  be a filename (key) and the contents of that file (value).

  See ReaderBase for supported methods.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  """

    @deprecation.deprecated(None, 'Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.map(tf.read_file)`.')
    def __init__(self, name=None):
        """Create a WholeFileReader.

    Args:
      name: A name for the operation (optional).
    """
        rr = gen_io_ops.whole_file_reader_v2(name=name)
        super(WholeFileReader, self).__init__(rr, supports_serialize=True)