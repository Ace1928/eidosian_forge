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
def _install_threaded_pg():
    global _old_pg_world
    global _ctx_manager
    _old_pg_world = dist.distributed_c10d._world
    dist.distributed_c10d._world = ThreadLocalWorld()
    _ctx_manager = torch.autograd.set_multithreading_enabled(False)
    return dist.distributed_c10d._world