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
def _get_nccl_p2p_communicator(self, comm_key, my_gpu_idx, peer_rank, peer_gpu_idx):
    """Create or retrieve an NCCL communicator for p2p tasks.

        Note(Hao): this function is not thread-safe now.

        Args:
            comm_key: communicator key.
            my_gpu_idx: the gpu index on the current process.
            peer_rank: the rank of the destination process.
            peer_gpu_idx: the gpu index on the peer process.
        Returns:
            communicator
        """
    if not comm_key:
        raise RuntimeError('Got empty communicator key.')
    if comm_key in self._dev_comm_map:
        return self._dev_comm_map[comm_key]
    if self.rank < peer_rank:
        my_p2p_rank = 0
    elif self.rank > peer_rank:
        my_p2p_rank = 1
    else:
        raise RuntimeError('Send and recv happens on the same process! ray.util.collective does not support this case as of now. Alternatively, consider doing GPU to GPU memcpy?')
    group_key = self._generate_group_key(comm_key)
    if my_p2p_rank == 0:
        nccl_uid = self._generate_nccl_uid(group_key)
    else:
        rendezvous = Rendezvous(group_key)
        rendezvous.meet()
        nccl_uid = rendezvous.get_nccl_id()
    with nccl_util.Device(my_gpu_idx):
        comm = nccl_util.create_nccl_communicator(2, nccl_uid, my_p2p_rank)
        stream = get_stream_pool(my_gpu_idx).get_stream()
        event = cupy.cuda.Event()
    self._dev_comm_map[comm_key] = [comm]
    self._dev_streams_map[comm_key] = [stream]
    self._dev_event_map[comm_key] = [event]
    return [comm]