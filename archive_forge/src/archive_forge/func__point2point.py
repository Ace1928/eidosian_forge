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
def _point2point(self, tensors, p2p_fn, peer_rank: int, peer_gpu_idx: int):
    """A method to encapsulate all peer-to-peer calls (i.e., send/recv).

        Args:
            tensors: the tensor to send or receive.
            p2p_fn: the p2p function call.
            peer_rank: the rank of the peer process.
            peer_gpu_idx: the index of the gpu on the peer process.

        Returns:
            None
        """
    if nccl_util.get_nccl_runtime_version() < 2704:
        raise RuntimeError("P2p send/recv requires NCCL >= 2.7.4. Got '{}'.".format(nccl_util.get_nccl_runtime_version()))
    _check_gpu_tensors(tensors)
    assert len(tensors) == 1
    my_gpu_idx = nccl_util.get_tensor_device(tensors[0])
    comm_key = _get_comm_key_send_recv(self.rank, my_gpu_idx, peer_rank, peer_gpu_idx)
    comms = self._get_nccl_p2p_communicator(comm_key, my_gpu_idx, peer_rank, peer_gpu_idx)
    streams = self._dev_streams_map[comm_key]
    events = self._dev_event_map[comm_key]
    self._sync_streams([my_gpu_idx], events, streams)
    peer_p2p_rank = 0 if self.rank > peer_rank else 1
    for i, tensor in enumerate(tensors):
        p2p_fn(tensors[i], comms[i], streams[i], peer_p2p_rank)