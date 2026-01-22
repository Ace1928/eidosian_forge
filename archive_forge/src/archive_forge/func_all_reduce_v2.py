from tensorflow.python.ops import gen_collective_ops
def all_reduce_v2(t, group_size, group_key, instance_key, merge_op='Add', final_op='Id', communication_hint='auto', timeout=0, ordering_token=None, max_subdivs_per_device=-1, name=None):
    """Reduces tensors collectively, across devices.

  Args:
    t: the tensor to be reduced.
    group_size: an int32 tensor. The total number of tensors to be collectively
      reduced.  Each must reside on a different device.  Should be a positive
      integer.
    group_key: an int32 tensor identifying the group of devices.
    instance_key: an int32 tensor identifying the participating group of Ops.
    merge_op: string naming the binary Op to be applied to compute each partial
      reduction.
    final_op: string naming the unary Op to be applied to each fully reduced
      value.  Can be 'Id' for no operation.
    communication_hint: preferred collective communication.  The implementation
      may fall back to another mechanism.  Options include `auto`, `ring`, and
      `nccl`.
    timeout: a float. If set to a non zero, set a completion timeout to detect
      staleness.  If the timer goes off, a DeadlineExceededError is raised.  The
      timeout value in seconds. This feature is experimental.
    ordering_token: a resource tensor on the same device as the op to order
      the collectives in a per-device manner by auto control dependency.
      This argument can be omited when there is one collective Op per
      `tf.function`, or when explicit control dependency is used instead of
      auto control dependency.
    max_subdivs_per_device: int specifying the maximum number of subdivisions a
      tensor on a device can be divided into. The runtime uses this contraint to
      parallelize processing of each per-device tensor. Setting to -1 disables
      subdivision and reverts to previous behavior of not sub-dividing tensor.
      Setting to 0 uses sytem defaults.
    name: name of the Op.

  Returns:
    An Op implementing the distributed reduction.
  """
    if ordering_token is not None:
        ordering_token = [ordering_token]
    else:
        ordering_token = []
    return gen_collective_ops.collective_reduce_v2(t, group_size=group_size, group_key=group_key, instance_key=instance_key, merge_op=merge_op, final_op=final_op, communication_hint=communication_hint.lower(), timeout_seconds=timeout, ordering_token=ordering_token, max_subdivs_per_device=max_subdivs_per_device, name=name)