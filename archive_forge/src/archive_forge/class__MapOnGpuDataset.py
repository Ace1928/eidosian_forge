from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import structure
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device as framework_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util.tf_export import tf_export
class _MapOnGpuDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that maps a function over elements in its using a GPU."""

    def __init__(self, input_dataset, map_func, use_inter_op_parallelism=True):
        """See `Dataset.map()` for details."""
        self._input_dataset = input_dataset
        self._use_inter_op_parallelism = use_inter_op_parallelism
        self._map_func = structured_function.StructuredFunctionWrapper(map_func, self._transformation_name(), dataset=input_dataset, defun_kwargs={'experimental_ints_on_device': True})
        variant_tensor = ged_ops.experimental_map_dataset(self._input_dataset._variant_tensor, self._map_func.function.captured_inputs, f=self._map_func.function, use_inter_op_parallelism=self._use_inter_op_parallelism, **self._flat_structure)
        super(_MapOnGpuDataset, self).__init__(input_dataset, variant_tensor)

    def _functions(self):
        return [self._map_func]

    @property
    def element_spec(self):
        return self._map_func.output_structure

    def _transformation_name(self):
        return 'map_on_gpu()'