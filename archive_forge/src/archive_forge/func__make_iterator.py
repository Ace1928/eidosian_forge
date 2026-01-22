from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import input_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util.deprecation import deprecated
def _make_iterator(self):
    """Make appropriate iterator on the dataset."""
    with ops.device(self._worker):
        if self._options is not None:
            self._iterator = multi_device_iterator_ops.MultiDeviceIterator(self._dataset, self._devices, max_buffer_size=self._options.experimental_per_replica_buffer_size, prefetch_buffer_size=self._options.experimental_per_replica_buffer_size)
        else:
            self._iterator = multi_device_iterator_ops.MultiDeviceIterator(self._dataset, self._devices)