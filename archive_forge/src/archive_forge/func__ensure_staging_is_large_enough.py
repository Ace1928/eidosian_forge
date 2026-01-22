import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, overload
import torch
import torch.distributed as dist
import torch.multiprocessing.reductions
from .. import _is_triton_available
from .common import BaseOperator, get_xformers_operator, register_operator
from .ipc import init_ipc
def _ensure_staging_is_large_enough(self, num_elements: int, random_init: bool):
    if self.staging.numel() < self.world_size * num_elements:
        self.staging = torch.empty((self.num_stripes, self.world_size, num_elements), device=self.my_device, dtype=self.dtype)
        if random_init:
            self.staging.normal_()
        for rank, conn in enumerate(self.p2p_comms):
            if conn is not None:
                conn.send(self.staging[:, rank])
        self.buddys_staging = [torch.empty((0,), device=self.my_device) if conn is None else conn.recv() for rank, conn in enumerate(self.p2p_comms)]