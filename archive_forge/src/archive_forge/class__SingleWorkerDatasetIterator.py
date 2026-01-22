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
class _SingleWorkerDatasetIterator(input_lib._SingleWorkerDatasetIteratorBase):
    """Iterator for a single DistributedDatasetV1 instance."""

    def _make_iterator(self):
        """Make appropriate iterator on the dataset."""
        with ops.device(self._worker):
            if self._options is not None:
                self._iterator = multi_device_iterator_ops.MultiDeviceIterator(self._dataset, self._devices, max_buffer_size=self._options.experimental_per_replica_buffer_size, prefetch_buffer_size=self._options.experimental_per_replica_buffer_size)
            else:
                self._iterator = multi_device_iterator_ops.MultiDeviceIterator(self._dataset, self._devices)

    def initialize(self):
        """Initialize underlying iterator.

    In eager execution, this simply recreates the underlying iterator.
    In graph execution, it returns the initializer ops for the underlying
    iterator.

    Returns:
      A list of any initializer ops that should be run.
    """
        if ops.executing_eagerly_outside_functions():
            self._iterator._eager_reset()
            return []
        else:
            return [self._iterator.initializer]

    @property
    def output_classes(self):
        return dataset_ops.get_legacy_output_classes(self._iterator)

    @property
    def output_shapes(self):
        return dataset_ops.get_legacy_output_shapes(self._iterator)

    @property
    def output_types(self):
        return dataset_ops.get_legacy_output_types(self._iterator)