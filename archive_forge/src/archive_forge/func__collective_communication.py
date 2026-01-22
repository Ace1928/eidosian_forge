import json
import os
import tensorflow.compat.v2 as tf
def _collective_communication(all_reduce_alg):
    """Return a CollectiveCommunication based on all_reduce_alg.

    Args:
      all_reduce_alg: a string specifying which collective communication to
        pick, or None.

    Returns:
      tf.distribute.experimental.CollectiveCommunication object

    Raises:
      ValueError: if `all_reduce_alg` not in [None, "ring", "nccl"]
    """
    collective_communication_options = {None: tf.distribute.experimental.CollectiveCommunication.AUTO, 'ring': tf.distribute.experimental.CollectiveCommunication.RING, 'nccl': tf.distribute.experimental.CollectiveCommunication.NCCL}
    if all_reduce_alg not in collective_communication_options:
        raise ValueError('When used with `multi_worker_mirrored`, valid values for all_reduce_alg are [`ring`, `nccl`].  Supplied value: {}'.format(all_reduce_alg))
    return collective_communication_options[all_reduce_alg]