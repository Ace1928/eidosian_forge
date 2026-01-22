import concurrent.futures
import json
import multiprocessing.connection
from typing import Any, List, Optional, Union
import torch
import torch.distributed as dist
import torch.multiprocessing.reductions
def _exchange_addresses(listeners: List[multiprocessing.connection.Listener], group: dist.ProcessGroup, device: torch.device) -> List[List[str]]:
    rank = group.rank()
    world_size = group.size()
    my_addresses: List[str] = []
    for listener in listeners:
        addr = listener.address
        if isinstance(addr, bytes):
            addr = addr.decode('utf-8')
        assert isinstance(addr, str)
        my_addresses.append(addr)
    if world_size == 1:
        return [my_addresses]
    try:
        _, store = torch.distributed.distributed_c10d._world.pg_map.get(group, (None, None))
        assert store is not None
        store.set(f'xformers_exchange_addresses_{rank}', json.dumps(my_addresses))
        all_addresses = [json.loads(store.get(f'xformers_exchange_addresses_{i}')) for i in range(world_size)]
    except Exception:
        all_addresses = [[''] * (world_size - 1)] * world_size
        with concurrent.futures.ThreadPoolExecutor(initializer=torch.cuda.set_device, initargs=(device,)) as e:
            e.submit(dist.all_gather_object, object_list=all_addresses, obj=my_addresses, group=group).result()
    return all_addresses