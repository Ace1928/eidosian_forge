import functools
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import random_op
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(None, 'Use `tf.data.Dataset.random(...)`.')
@tf_export('data.experimental.RandomDataset', v1=[])
class RandomDatasetV2(random_op._RandomDataset):
    """A `Dataset` of pseudorandom values."""