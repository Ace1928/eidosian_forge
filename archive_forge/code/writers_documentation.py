from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import convert
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
Writes a dataset to a TFRecord file.

    An operation that writes the content of the specified dataset to the file
    specified in the constructor.

    If the file exists, it will be overwritten.

    Args:
      dataset: a `tf.data.Dataset` whose elements are to be written to a file

    Returns:
      In graph mode, this returns an operation which when executed performs the
      write. In eager mode, the write is performed by the method itself and
      there is no return value.

    Raises
      TypeError: if `dataset` is not a `tf.data.Dataset`.
      TypeError: if the elements produced by the dataset are not scalar strings.
    