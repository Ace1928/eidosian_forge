import collections
import enum
from typing import cast, Dict, List, Set, Tuple
import torch
import torch.distributed as dist
from ._utils import _group_membership_management, _update_group_membership
from . import api
from . import constants as rpc_constants
def _tensorpipe_init_backend_handler(store, name, rank, world_size, rpc_backend_options):
    from . import TensorPipeAgent
    from . import TensorPipeRpcBackendOptions
    if not isinstance(store, dist.Store):
        raise TypeError(f'`store` must be a c10d::Store. {store}')
    if not isinstance(rpc_backend_options, TensorPipeRpcBackendOptions):
        raise TypeError(f'`rpc_backend_options` must be a `TensorPipeRpcBackendOptions`. {rpc_backend_options}')
    device_count = torch.cuda.device_count()
    is_static_group = True if world_size else False
    if is_static_group:
        group = _init_process_group(store, rank, world_size)
        reverse_device_maps, devices = _tensorpipe_exchange_and_check_all_device_maps(name, device_count, rpc_backend_options.device_maps, rpc_backend_options.devices, group)
        if torch.cuda.is_available() and devices:
            torch.cuda.init()
        agent = TensorPipeAgent(store, name, rank, world_size, rpc_backend_options, reverse_device_maps, devices)
        api._init_rpc_states(agent)
        api._all_gather(None, timeout=rpc_backend_options.rpc_timeout)
        group.barrier().wait()
        return agent
    else:
        with _group_membership_management(store, name, True):
            agent = TensorPipeAgent(store, name, rank, world_size, rpc_backend_options, {}, [])
            api._init_rpc_states(agent)
            try:
                _set_devices_and_reverse_device_map(agent)
                pass
            except Exception:
                api.shutdown()
                raise
            return agent