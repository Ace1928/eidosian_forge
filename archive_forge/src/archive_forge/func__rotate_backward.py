from abc import ABC, abstractmethod
from math import log as mlog
from typing import List, Optional, Tuple
import torch
import torch.distributed as dist
def _rotate_backward(self, r: int, p: int) -> int:
    """Helper function returns peer that is p hops behind r"""
    return (r - p) % self.world_size