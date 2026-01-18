from abc import ABC, abstractmethod
import inspect
from typing import Dict, Type
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.distributed.optim import as_functional_optim
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.distributed.algorithms.ddp_comm_hooks.optimizer_overlap_hooks import (
Register the overlapped optimizer with FSDP.