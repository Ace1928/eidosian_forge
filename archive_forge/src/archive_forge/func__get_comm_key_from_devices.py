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
def _get_comm_key_from_devices(devices):
    """Return a key from a list of devices for collective calls.

    For example, if the tensors are on gpus 0, 1, 2, 3,
    then the key would be "0,1,2,3".

    Args:
        devices: a list of GPU device indices

    Returns:
        str: a string represents the key to query the communicator cache.

    """
    return ','.join([str(d) for d in devices])