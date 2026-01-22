import contextlib
import threading
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
Returns the current worker index, when called within a worker closure.

  Some parameter server training workloads may require the worker to know its
  index, for example for data sharding for reduced-variance training.

  This method may be used within a `tf.function` that is executed on a worker.
  That is, either a `dataset_fn` that runs via
  `ClusterCoordinator.create_per_worker_dataset`, or any other function
  scheduled via `ClusterCoordinator.schedule`.

  Example (sharding data by worker):

  ```python
  strategy = tf.distribute.ParameterServerStrategy(
      cluster_resolver=...)
  coordinator = (
      tf.distribute.coordinator.ClusterCoordinator(strategy))

  def dataset_fn(context):
    dataset = tf.data.Dataset.range(10)
    worker_index = (
        tf.distribute.coordinator.experimental_get_current_worker_index()
    )
    dataset = dataset.shard(
        num_shards=num_workers,
        index=worker_index,
    )
    return dataset

  @tf.function
  def per_worker_dataset_fn():
    return strategy.distribute_datasets_from_function(dataset_fn)

  per_worker_dataset = coordinator.create_per_worker_dataset(
      per_worker_dataset_fn)
  ```

  Raises:
    RuntimeError: if called from outside a `tf.function` or outside of a remote
      closure execution context (that is, on a non-worker machine).
  