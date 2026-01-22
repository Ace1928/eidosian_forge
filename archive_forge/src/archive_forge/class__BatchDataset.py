import warnings
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_dataset_ops
class _BatchDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that batches contiguous elements from its input."""

    def __init__(self, input_dataset, batch_size, drop_remainder, name=None):
        """See `Dataset.batch()` for details."""
        self._input_dataset = input_dataset
        self._batch_size = ops.convert_to_tensor(batch_size, dtype=dtypes.int64, name='batch_size')
        self._drop_remainder = ops.convert_to_tensor(drop_remainder, dtype=dtypes.bool, name='drop_remainder')
        constant_drop_remainder = tensor_util.constant_value(self._drop_remainder)
        if constant_drop_remainder:
            constant_batch_size = tensor_util.constant_value(self._batch_size)
            self._structure = nest.map_structure(lambda component_spec: component_spec._batch(constant_batch_size), input_dataset.element_spec)
        else:
            self._structure = nest.map_structure(lambda component_spec: component_spec._batch(None), input_dataset.element_spec)
        self._name = name
        variant_tensor = gen_dataset_ops.batch_dataset_v2(input_dataset._variant_tensor, batch_size=self._batch_size, drop_remainder=self._drop_remainder, **self._common_args)
        super().__init__(input_dataset, variant_tensor)

    @property
    def element_spec(self):
        return self._structure