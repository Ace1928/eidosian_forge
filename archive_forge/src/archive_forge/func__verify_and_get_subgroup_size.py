from tensorflow.python.framework import ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export
def _verify_and_get_subgroup_size(self, group_assignment, num_shards):
    """Verify group_assignment and get the subgroup size".

    Args:
      group_assignment: list of group ids for applying the optimizer
        to subgroups.
      num_shards: The number of TPU shards.

    Returns:
      The size of one subgroup in group_assignment.

    Raises:
      ValueError: If group_assignment is invalid.
    """
    if not group_assignment:
        return None
    if not (isinstance(group_assignment, list) and all((isinstance(i, list) for i in group_assignment))):
        raise ValueError(f'Argument `group_assignment` must be a list of lists. Received: {group_assignment}')
    replica_ids = set()
    for g in group_assignment:
        for i in g:
            replica_ids.add(i)
    if set(range(num_shards)) != replica_ids:
        raise ValueError(f'Argument `group_assignment` must be a permutation of range({num_shards}). Received: {group_assignment}')
    subgroup_size_list = [len(group) for group in group_assignment]
    if all((subgroup_size_list[0] == size for size in subgroup_size_list)):
        return subgroup_size_list[0]
    else:
        raise ValueError(f'The size of each subgroup in `group_assignment` must be equal. Received: {group_assignment}')