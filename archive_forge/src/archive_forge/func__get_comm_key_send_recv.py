import logging
import datetime
import time
import ray
import cupy
from ray.util.collective.const import ENV
from ray.util.collective.collective_group import nccl_util
from ray.util.collective.collective_group.base_collective_group import BaseGroup
from ray.util.collective.const import get_store_name
from ray.util.collective.types import (
from ray.util.collective.collective_group.cuda_stream import get_stream_pool
def _get_comm_key_send_recv(my_rank, my_gpu_idx, peer_rank, peer_gpu_idx):
    """Return a key given source and destination ranks for p2p tasks.

    The p2p key is in the following form:
                [min_rank]_[gpu_index]:[max_rank]_[gpu_index].

    Args:
        my_rank: the rank of the source process.
        my_gpu_idx: the source gpu index on the process.
        peer_rank: the rank of the destination process.
        peer_gpu_idx: the destination gpu index on the process.

    Returns:
        comm_key: a string key to query the communication cache.
    """
    if my_rank < peer_rank:
        lower_key = str(my_rank) + '_' + str(my_gpu_idx)
        higher_key = str(peer_rank) + '_' + str(peer_gpu_idx)
    elif my_rank > peer_rank:
        lower_key = str(peer_rank) + '_' + str(peer_gpu_idx)
        higher_key = str(my_rank) + '_' + str(my_gpu_idx)
    else:
        raise RuntimeError('Send and recv happens on the same process. ray.util.collective does not support this case as of now. Alternatively, consider doing GPU to GPU memcpy?')
    comm_key = lower_key + ':' + higher_key
    return comm_key