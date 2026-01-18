import collections
import copy
import enum
import inspect
import io
import logging
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.distributed.optim.utils import functional_optim_map
from torch.optim import Optimizer
def wait_for_broadcasts(self) -> None:
    """
        Wait for all parameter broadcasts.

        This function should be called once all broadcasts have been scheduled,
        meaning ``self.broadcast_handles`` is filled. This clears ``self.broadcast_handles``
        in preparation for the next iteration.
        """
    assert len(self.broadcast_handles) == self.num_bucket_assignments, f'Missing at least one broadcast handle on rank {dist.get_rank()}'
    _ = [x.wait() for x in self.broadcast_handles]
    self.broadcast_handles.clear()