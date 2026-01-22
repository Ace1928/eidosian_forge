from contextlib import contextmanager
from typing import cast
import logging
from . import api
from . import TensorPipeAgent
def _update_group_membership(worker_info, my_devices, reverse_device_map, is_join):
    agent = cast(TensorPipeAgent, api._get_current_rpc_agent())
    ret = agent._update_group_membership(worker_info, my_devices, reverse_device_map, is_join)
    return ret