import abc
import functools
import json
import os
import signal
import socket
import time
import traceback
import warnings
from contextlib import closing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch.distributed.elastic.rendezvous as rdzv
import torch.distributed.elastic.utils.store as store_util
from torch.distributed import Store
from torch.distributed.elastic.events import Event, EventSource, record
from torch.distributed.elastic.metrics import prof, put_metric
from torch.distributed.elastic.multiprocessing import (
from torch.distributed.elastic.utils.logging import get_logger
@prof
def _assign_worker_ranks(self, store, group_rank: int, group_world_size: int, spec: WorkerSpec) -> List[Worker]:
    """Determine proper ranks for worker processes.

        The rank assignment is done according to the following algorithm:

        1. Each agent writes its configuration(group_rank, group_world_size
           , num_workers) to the common store.
        2. Each agent retrieves configuration for all agents
           and performs two level sort using role and rank.
        3. Determine the global rank: the global rank of the workers for the current
           agent is the offset of the infos array up to group_rank of the agent.
           The offset is computed as a sum of local_world_size of all agents that
           have rank less than the group_rank. The workers would have the ranks:
           [offset, offset+local_world_size)
        4. Determine the role rank: The role rank is determined using the algorithms
           in the point 3 with the exception that the offset is done from the first
           agent that has the same role as current one and has the minimum group rank.
        """
    role_infos = self._share_and_gather(store, group_rank, group_world_size, spec)
    my_role_info = role_infos[group_rank]
    worker_world_size, worker_global_ranks = self._get_ranks(role_infos, group_rank)
    role_infos = sorted(role_infos, key=functools.cmp_to_key(_RoleInstanceInfo.compare))
    role_start_idx, role_end_idx = _RoleInstanceInfo.find_role_boundaries(role_infos, my_role_info.role)
    role_pos = next((idx for idx, role_info in enumerate(role_infos) if _RoleInstanceInfo.compare(role_info, my_role_info) == 0))
    role_world_size, role_ranks = self._get_ranks(role_infos, role_pos, role_start_idx, role_end_idx + 1)
    workers = []
    for ind in range(spec.local_world_size):
        worker = Worker(local_rank=ind, global_rank=worker_global_ranks[ind], role_rank=role_ranks[ind], world_size=worker_world_size, role_world_size=role_world_size)
        workers.append(worker)
    return workers