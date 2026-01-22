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
class DatasetIterator(DistributedIteratorV1):
    """Iterator created from input dataset."""

    def __init__(self, dataset, input_workers, strategy, num_replicas_in_sync=None, input_context=None):
        """Make an iterator for the dataset on given devices.

    If `num_replicas_in_sync` is not None, we split each batch of the dataset
    into `num_replicas_in_sync` smaller batches, to be distributed among that
    worker's replicas, so that the batch size for a global step (across all
    workers and replicas) is as expected.

    Args:
      dataset: `tf.data.Dataset` that will be used as the input source.
      input_workers: an `InputWorkers` object.
      strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
        handle last partial batch.
      num_replicas_in_sync: Optional integer. If this is not None, the value is
        used to decide how to rebatch datasets into smaller batches so that the
        total batch size for each step (across all workers and replicas) adds up
        to `dataset`'s batch size.
      input_context: `InputContext` for sharding. Only pass this in for between
        graph multi-worker cases where there is only one `input_worker`. In
        these cases, we will shard based on the `input_pipeline_id` and
        `num_input_pipelines` in the `InputContext`.
    """
        dist_dataset = DistributedDatasetV1(dataset, input_workers, strategy, num_replicas_in_sync=num_replicas_in_sync, input_context=input_context)
        worker_iterators = _create_iterators_per_worker(dist_dataset._cloned_datasets, input_workers)
        super(DatasetIterator, self).__init__(input_workers, worker_iterators, strategy, dist_dataset.cardinality, dist_dataset._enable_get_next_as_optional)
        self._element_spec = dist_dataset.element_spec