import os
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import from_tensor_slices_op
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import convert
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
class ParallelInterleaveDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that maps a function over its input and flattens the result."""

    def __init__(self, input_dataset, map_func, cycle_length, block_length, sloppy, buffer_output_elements, prefetch_input_elements, name=None):
        """See `tf.data.experimental.parallel_interleave()` for details."""
        self._input_dataset = input_dataset
        self._map_func = structured_function.StructuredFunctionWrapper(map_func, self._transformation_name(), dataset=input_dataset)
        if not isinstance(self._map_func.output_structure, dataset_ops.DatasetSpec):
            raise TypeError(f'The `map_func` argument must return a `Dataset` object. Got {_get_type(self._map_func.output_structure)!r}.')
        self._element_spec = self._map_func.output_structure._element_spec
        self._cycle_length = ops.convert_to_tensor(cycle_length, dtype=dtypes.int64, name='cycle_length')
        self._block_length = ops.convert_to_tensor(block_length, dtype=dtypes.int64, name='block_length')
        self._buffer_output_elements = convert.optional_param_to_tensor('buffer_output_elements', buffer_output_elements, argument_default=2 * block_length)
        self._prefetch_input_elements = convert.optional_param_to_tensor('prefetch_input_elements', prefetch_input_elements, argument_default=2 * cycle_length)
        if sloppy is None:
            self._deterministic = 'default'
        elif sloppy:
            self._deterministic = 'false'
        else:
            self._deterministic = 'true'
        self._name = name
        variant_tensor = ged_ops.legacy_parallel_interleave_dataset_v2(self._input_dataset._variant_tensor, self._map_func.function.captured_inputs, self._cycle_length, self._block_length, self._buffer_output_elements, self._prefetch_input_elements, f=self._map_func.function, deterministic=self._deterministic, **self._common_args)
        super(ParallelInterleaveDataset, self).__init__(input_dataset, variant_tensor)

    def _functions(self):
        return [self._map_func]

    @property
    def element_spec(self):
        return self._element_spec

    def _transformation_name(self):
        return 'tf.data.experimental.parallel_interleave()'