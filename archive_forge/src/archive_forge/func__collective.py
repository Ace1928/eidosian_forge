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
def _collective(self, input_tensors, output_tensors, collective_fn, preprocess_fn=None, postprocess_fn=None):
    """A method to encapsulate all collective calls.

        Args:
            input_tensors: the list of the input tensors.
            output_tensors: the list of the output tensors.
            collective_fn: the collective function call.
            preprocess_fn: preprocess procedures before collective calls.
            postprocess_fn: postprocess procedures after collective calls.

        Returns:
            None
        """
    _check_gpu_tensors(input_tensors)
    _check_gpu_tensors(output_tensors)
    devices = nccl_util.get_tensor_device_list(input_tensors)
    key = _get_comm_key_from_devices(devices)
    comms = self._get_nccl_collective_communicator(key, devices)
    streams = self._dev_streams_map[key]
    events = self._dev_event_map[key]
    self._sync_streams(devices, events, streams)
    if preprocess_fn:
        preprocess_fn(streams)
    nccl_util.groupStart()
    for i, tensor in enumerate(input_tensors):
        collective_fn(tensor, output_tensors[i], comms[i], streams[i])
    nccl_util.groupEnd()
    if postprocess_fn:
        postprocess_fn(streams)