import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
from sympy import Integer
from .. import metrics
from ..scheduler import SchedulerNode
from ..utils import ceildiv, Placeholder
from ..virtualized import V
from .common import IndentedBuffer, Kernel
from .triton import TritonKernel
from .triton_utils import config_of, signature_to_meta
@dataclass
class PartitionState:
    partitions: List[List[Tuple[List[SchedulerNode], Tuple[Integer, ...], Integer, Integer]]]
    cur_partition: List[Tuple[List[SchedulerNode], Tuple[Integer, ...], Integer, Integer]]
    cur_count: int

    def finalize(self):
        if self.cur_partition:
            self.partitions.append(self.cur_partition)