import collections
import enum
from typing import cast, Dict, List, Set, Tuple
import torch
import torch.distributed as dist
from ._utils import _group_membership_management, _update_group_membership
from . import api
from . import constants as rpc_constants
def _create_device_list(my_devices, my_device_maps, reverse_device_maps):
    if not my_devices:
        devices_set: Set[torch.device] = set()
        for map_ in my_device_maps.values():
            devices_set.update(map_.keys())
        for map_ in reverse_device_maps.values():
            devices_set.update(map_.keys())
        devices_set.discard(torch.device('cpu'))
        my_devices = list(devices_set)
    my_devices = sorted(my_devices, key=lambda d: d.index)
    return my_devices