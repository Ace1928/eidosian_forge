import logging
import warnings
from copy import deepcopy
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Union, overload
import torch
import torch.nn as nn
from torch import optim
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        Run a dummy optimizer step, which allows to initialize optimizer state because we do lazy init for most optimizers.

        This allows doing in-place loading of optimizer state from a checkpoint.
        