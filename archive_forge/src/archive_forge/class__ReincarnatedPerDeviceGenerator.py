from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import prefetch_op
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import resource_variable_ops
class _ReincarnatedPerDeviceGenerator(dataset_ops.DatasetV2):
    """Creates a _PerDeviceGenerator-like dataset with a new incarnation_id.

  Re-uses the functions from the provided per_device_dataset and just switches
  out the function argument corresponding to the incarnation_id.
  """

    def __init__(self, per_device_dataset, incarnation_id):
        self._element_spec = per_device_dataset.element_spec
        self._init_func = per_device_dataset._init_func
        self._init_captured_args = self._init_func.captured_inputs
        self._next_func = per_device_dataset._next_func
        self._next_captured_args = per_device_dataset._next_captured_args
        self._next_captured_args[per_device_dataset._incarnation_id_index] = incarnation_id
        self._finalize_func = per_device_dataset._finalize_func
        self._finalize_captured_args = per_device_dataset._finalize_captured_args
        variant_tensor = gen_dataset_ops.generator_dataset(self._init_captured_args, self._next_captured_args, self._finalize_captured_args, init_func=self._init_func, next_func=self._next_func, finalize_func=self._finalize_func, **self._flat_structure)
        super(_ReincarnatedPerDeviceGenerator, self).__init__(variant_tensor)

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._element_spec