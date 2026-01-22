from tensorflow.python.ops import gen_collective_ops
Exchanges tensors mutually.

  Args:
    communicator: the resource `tf.Tensor` returned from
      `initialize_communicator`.
    t: a `tf.Tensor`. The first dimension should have the length as the size of
      the group. `t[i]` is sent to `rank i` within the group.
    group_assignment: Optional int32 `tf.Tensor` with shape [num_groups,
      num_ranks_per_group]. `group_assignment[i]` represents the ranks in the
      `ith` subgroup.
    timeout_seconds: If set to a non zero, set a completion timeout to detect
      staleness. If the timer goes off, a DeadlineExceededError is raised. The
      timeout value in seconds. This feature is experimental.

  Returns:
    a `tf.Tensor`. `t[i]` is sent from `rank i` within the group.
  