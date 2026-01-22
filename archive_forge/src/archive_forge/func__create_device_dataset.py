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
def _create_device_dataset(prototype_ds, incarnation_id, prefetch_buffer_size, experimental_slack):
    """Uses _prototype_device_datasets[i] to build a dataset for the device."""
    ds = _ReincarnatedPerDeviceGenerator(prototype_ds, incarnation_id)
    if prefetch_buffer_size > 0:
        if experimental_slack:
            ds = prefetch_op._PrefetchDataset(ds, prefetch_buffer_size, slack_period=1)
        else:
            ds = ds.prefetch(prefetch_buffer_size)
    return ds