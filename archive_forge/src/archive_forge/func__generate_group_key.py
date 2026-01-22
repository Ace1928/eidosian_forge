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
def _generate_group_key(self, comm_key):
    """Generate a unique key used to initialize the KV store.

        The group key is a concatenation of the communicator key and
        the group name, following: [comm_key]@[group_name].
        """
    return comm_key + '@' + self.group_name