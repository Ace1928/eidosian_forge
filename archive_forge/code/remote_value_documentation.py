import enum
from tensorflow.python.util.tf_export import tf_export
Wait for the result of `RemoteValue` and return the tensor result.

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
    