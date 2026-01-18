from collections.abc import Mapping
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
Pads a dataset with fake elements to reach the desired cardinality.

  The dataset to pad must have a known and finite cardinality and contain
  dictionary elements. The `mask_key` will be added to differentiate between
  real and padding elements -- real elements will have a `<mask_key>=True` entry
  while padding elements will have a `<mask_key>=False` entry.

  Example usage:

  >>> ds = tf.data.Dataset.from_tensor_slices({'a': [1, 2]})
  >>> ds = ds.apply(tf.data.experimental.pad_to_cardinality(3))
  >>> list(ds.as_numpy_iterator())
  [{'a': 1, 'valid': True}, {'a': 2, 'valid': True}, {'a': 0, 'valid': False}]

  This can be useful, e.g. during eval, when partial batches are undesirable but
  it is also important not to drop any data.

  ```
  ds = ...
  # Round up to the next full batch.
  target_cardinality = -(-ds.cardinality() // batch_size) * batch_size
  ds = ds.apply(tf.data.experimental.pad_to_cardinality(target_cardinality))
  # Set `drop_remainder` so that batch shape will be known statically. No data
  # will actually be dropped since the batch size divides the cardinality.
  ds = ds.batch(batch_size, drop_remainder=True)
  ```

  Args:
    cardinality: The cardinality to pad the dataset to.
    mask_key: The key to use for identifying real vs padding elements.

  Returns:
    A dataset transformation that can be applied via `Dataset.apply()`.
  