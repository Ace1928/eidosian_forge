import sys
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from functools import partial, reduce
import torch
import torch.distributed as dist
import weakref
from torch._C._distributed_c10d import (
from torch.distributed.distributed_c10d import _CollOp, _store_based_barrier, P2POp
from torch.futures import Future
from torch.utils import _pytree as pytree
@dataclass
class WorldData:
    default_pg: dist.ProcessGroup
    pg_map: Dict[dist.ProcessGroup, Tuple[str, Optional[Store]]]
    pg_names: Dict[dist.ProcessGroup, str]
    pg_group_ranks: Dict[dist.ProcessGroup, Dict[int, int]]
    pg_backend_config: Dict[dist.ProcessGroup, str]
    group_count: int
    tags_to_pg: Dict[str, List[dist.ProcessGroup]]
    pg_to_tag: Dict[dist.ProcessGroup, str]
    pg_coalesce_state: Dict[dist.ProcessGroup, List[Union[_CollOp, P2POp]]]
    pg_default_device: Dict[dist.ProcessGroup, torch.device]