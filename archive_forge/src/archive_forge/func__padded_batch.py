import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_dataset_ops
def _padded_batch(input_dataset, batch_size, padded_shapes=None, padding_values=None, drop_remainder=False, name=None):
    """See `tf.data.Dataset.padded_batch` for details."""
    if padded_shapes is None:
        padded_shapes = dataset_ops.get_legacy_output_shapes(input_dataset)
        for i, shape in enumerate(nest.flatten(padded_shapes)):
            if not shape:
                raise ValueError(f'You must provide `padded_shapes` argument because component {i} has unknown rank.')
    return _PaddedBatchDataset(input_dataset, batch_size, padded_shapes, padding_values, drop_remainder, name=name)