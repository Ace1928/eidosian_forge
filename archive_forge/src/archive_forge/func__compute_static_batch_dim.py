import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
def _compute_static_batch_dim(self):
    """Computes the static batch dimension of a dataset if it can be determined.

    Given the RebatchDataset parameters, determines the batch dimension of this
    dataset statically. Returns None if this cannot be determined or is
    variable.

    Returns:
      An integer representing the batch dimension of the dataset. If it cannot
      be determined statically, returns None.

    Raises:
      ValueError: The batch_sizes parameter is malformed, input_dataset is
      not batched, or input_dataset batch sizes are incompatible with each
      other.
    """
    new_batch_dim = tensor_util.constant_value(self._batch_sizes)
    if new_batch_dim is None:
        return None
    if isinstance(new_batch_dim, np.ndarray):
        if len(new_batch_dim.shape) == 1:
            if np.all(new_batch_dim == new_batch_dim[0]):
                new_batch_dim = new_batch_dim[0]
            else:
                return None
        elif len(new_batch_dim.shape) > 1:
            raise ValueError(f'Invalid `batch_sizes`. Expected `batch_sizes` to be a scalar or a vector. Received `batch_sizes` of rank {len(new_batch_dim.shape)}.')
    if self._may_form_partial_batches(new_batch_dim):
        return None
    return new_batch_dim