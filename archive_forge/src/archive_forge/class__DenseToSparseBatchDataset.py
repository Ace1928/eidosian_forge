from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import convert
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
class _DenseToSparseBatchDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that batches ragged dense elements into `tf.sparse.SparseTensor`s."""

    def __init__(self, input_dataset, batch_size, row_shape):
        """See `Dataset.dense_to_sparse_batch()` for more details."""
        if not isinstance(dataset_ops.get_legacy_output_types(input_dataset), dtypes.DType):
            raise TypeError(f'`dense_to_sparse_batch` requires an input dataset whose elements have a single component, but the given dataset has the following component types: {dataset_ops.get_legacy_output_types(input_dataset)}.')
        self._input_dataset = input_dataset
        self._batch_size = batch_size
        self._row_shape = row_shape
        self._element_spec = sparse_tensor.SparseTensorSpec(tensor_shape.TensorShape([None]).concatenate(self._row_shape), dataset_ops.get_legacy_output_types(input_dataset))
        variant_tensor = ged_ops.dense_to_sparse_batch_dataset(self._input_dataset._variant_tensor, self._batch_size, row_shape=convert.partial_shape_to_tensor(self._row_shape), **self._flat_structure)
        super(_DenseToSparseBatchDataset, self).__init__(input_dataset, variant_tensor)

    @property
    def element_spec(self):
        return self._element_spec