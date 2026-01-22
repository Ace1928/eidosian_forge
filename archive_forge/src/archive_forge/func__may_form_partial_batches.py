import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
def _may_form_partial_batches(self, desired_batch_size):
    """Returns whether this dataset may form partial batches."""
    if tensor_util.constant_value(self._drop_remainder):
        return False

    def get_batch_dim(type_spec):
        try:
            shape = type_spec._to_legacy_output_shapes()
        except NotImplementedError:
            return None
        if not isinstance(shape, tensor_shape.TensorShape):
            return None
        if shape.rank is None:
            return None
        if len(shape) < 1:
            raise ValueError('Invalid `batch_sizes`. Expected dataset with rank of >= 1 but found a dataset with scalar elements. Fix the issue by adding the `batch` transformation to the dataset.')
        return shape.dims[0].value
    input_batch_dims = [get_batch_dim(ts) for ts in nest.flatten(dataset_ops.get_structure(self._input_dataset))]
    known_input_batch_dims = [d for d in input_batch_dims if d is not None]
    if not known_input_batch_dims:
        return True
    known_input_batch_dims = np.asarray(known_input_batch_dims)
    if not np.all(known_input_batch_dims == known_input_batch_dims[0]):
        raise ValueError(f'Invalid `input_dataset.` The batch dimension of component 0 is {known_input_batch_dims[0]}, while the batch dimension of component i is {known_input_batch_dims}.')
    return known_input_batch_dims[0] % desired_batch_size != 0