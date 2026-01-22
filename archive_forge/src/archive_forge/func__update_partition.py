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
@staticmethod
def _update_partition(partition_state, node_rw_count, node_info):
    if partition_state.cur_count + node_rw_count > ForeachKernel.MAX_NUM_ARGS:
        partition_state.partitions.append(partition_state.cur_partition)
        partition_state.cur_partition = [node_info]
        partition_state.cur_count = node_rw_count
    else:
        partition_state.cur_count += node_rw_count
        partition_state.cur_partition.append(node_info)