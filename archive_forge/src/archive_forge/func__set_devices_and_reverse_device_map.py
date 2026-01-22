import collections
import enum
from typing import cast, Dict, List, Set, Tuple
import torch
import torch.distributed as dist
from ._utils import _group_membership_management, _update_group_membership
from . import api
from . import constants as rpc_constants
def _set_devices_and_reverse_device_map(agent):
    from . import TensorPipeAgent
    agent = cast(TensorPipeAgent, agent)
    my_worker_info = agent.get_worker_info()
    my_name = my_worker_info.name
    all_worker_infos = agent.get_worker_infos()
    all_device_counts, all_device_maps, all_devices, all_names = ({}, {}, {}, [])
    for worker_info in all_worker_infos:
        worker_name = worker_info.name
        if worker_name != my_name:
            device_count, device_map, devices = api.rpc_sync(worker_name, _get_device_infos)
        else:
            opts = agent._get_backend_options()
            device_count, device_map, devices = (torch.cuda.device_count(), opts.device_maps, opts.devices)
        all_device_counts[worker_name] = device_count
        all_device_maps[worker_name] = device_map
        all_devices[worker_name] = devices
        all_names.append(worker_name)
    _validate_device_maps(all_names, all_device_counts, all_device_maps, all_devices, is_static_group=False)
    reverse_device_maps = _create_reverse_mapping(my_name, all_names, all_device_maps)
    for worker_name in all_names:
        all_devices[worker_name] = _create_device_list(all_devices[worker_name], all_device_maps[worker_name], reverse_device_maps)
        api.rpc_sync(worker_name, _update_group_membership, args=(my_worker_info, all_devices[worker_name], reverse_device_maps, True))