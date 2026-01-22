import enum
from tensorflow.python.util.tf_export import tf_export
@tf_export('distribute.experimental.coordinator.RemoteValue', 'distribute.coordinator.RemoteValue', v1=[])
class RemoteValue(object):
    """An asynchronously available value of a scheduled function.

  This class is used as the return value of
  `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` where
  the underlying value becomes available at a later time once the function has
  been executed.

  Using `tf.distribute.experimental.coordinator.RemoteValue` as an input to
  a subsequent function scheduled with
  `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` is
  currently not supported.

  Example:

  ```python
  strategy = tf.distribute.experimental.ParameterServerStrategy(
      cluster_resolver=...)
  coordinator = (
      tf.distribute.experimental.coordinator.ClusterCoordinator(strategy))

  with strategy.scope():
    v1 = tf.Variable(initial_value=0.0)
    v2 = tf.Variable(initial_value=1.0)

  @tf.function
  def worker_fn():
    v1.assign_add(0.1)
    v2.assign_sub(0.2)
    return v1.read_value() / v2.read_value()

  result = coordinator.schedule(worker_fn)
  # Note that `fetch()` gives the actual result instead of a `tf.Tensor`.
  assert result.fetch() == 0.125

  for _ in range(10):
    # `worker_fn` will be run on arbitrary workers that are available. The
    # `result` value will be available later.
    result = coordinator.schedule(worker_fn)
  ```
  """

    def fetch(self):
        """Wait for the result of `RemoteValue` and return the numpy result.

    This makes the value concrete by copying the remote value to local.

    Returns:
      The numpy array structure of the actual output of the `tf.function`
      associated with this `RemoteValue`, previously returned by a
      `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` call.
      This can be a single value, or a structure of values, depending on the
      output of the `tf.function`.

    Raises:
      tf.errors.CancelledError: If the function that produces this `RemoteValue`
        is aborted or cancelled due to failure.
    """
        raise NotImplementedError('Must be implemented in subclasses.')

    def get(self):
        """Wait for the result of `RemoteValue` and return the tensor result.

    This makes the value concrete by copying the remote tensor to local.

    Returns:
      The actual output (in the form of `tf.Tensor`s) of the `tf.function`
      associated with this `RemoteValue`, previously returned by a
      `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` call.
      This can be a single Tensor, or a structure of Tensors, depending on the
      output of the `tf.function`.

    Raises:
      tf.errors.CancelledError: If the function that produces this `RemoteValue`
        is aborted or cancelled due to failure.
    """
        raise NotImplementedError('Must be implemented in subclasses.')