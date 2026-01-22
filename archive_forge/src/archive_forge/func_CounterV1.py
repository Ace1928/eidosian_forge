from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['data.experimental.Counter'])
@deprecation.deprecated(None, 'Use `tf.data.Dataset.counter(...)` instead.')
def CounterV1(start=0, step=1, dtype=dtypes.int64):
    return dataset_ops.DatasetV1Adapter(CounterV2(start, step, dtype))