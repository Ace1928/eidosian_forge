import collections
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4
import numpy as np
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces.op_runtime_metrics import OpRuntimeMetrics
from ray.data._internal.util import capfirst
from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.util.annotations import DeveloperAPI
from ray.util.metrics import Gauge
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
@dataclass
class IterStatsSummary:
    wait_time: Timer
    get_time: Timer
    next_time: Timer
    format_time: Timer
    collate_time: Timer
    finalize_batch_time: Timer
    block_time: Timer
    user_time: Timer
    total_time: Timer
    iter_blocks_local: int
    iter_blocks_remote: int
    iter_unknown_location: int

    def __str__(self) -> str:
        return self.to_string()

    def to_string(self) -> str:
        out = ''
        if self.block_time.get() or self.total_time.get() or self.get_time.get() or self.next_time.get() or self.format_time.get() or self.collate_time.get() or self.finalize_batch_time.get():
            out += '\nDataset iterator time breakdown:\n'
            if self.block_time.get():
                out += '* Total time user code is blocked: {}\n'.format(fmt(self.block_time.get()))
            if self.user_time.get():
                out += '* Total time in user code: {}\n'.format(fmt(self.user_time.get()))
            if self.total_time.get():
                out += '* Total time overall: {}\n'.format(fmt(self.total_time.get()))
            out += '* Num blocks local: {}\n'.format(self.iter_blocks_local)
            out += '* Num blocks remote: {}\n'.format(self.iter_blocks_remote)
            out += '* Num blocks unknown location: {}\n'.format(self.iter_unknown_location)
            out += '* Batch iteration time breakdown (summed across prefetch threads):\n'
            if self.get_time.get():
                out += '    * In ray.get(): {} min, {} max, {} avg, {} total\n'.format(fmt(self.get_time.min()), fmt(self.get_time.max()), fmt(self.get_time.avg()), fmt(self.get_time.get()))
            if self.next_time.get():
                batch_creation_str = '    * In batch creation: {} min, {} max, {} avg, {} total\n'
                out += batch_creation_str.format(fmt(self.next_time.min()), fmt(self.next_time.max()), fmt(self.next_time.avg()), fmt(self.next_time.get()))
            if self.format_time.get():
                format_str = '    * In batch formatting: {} min, {} max, {} avg, {} total\n'
                out += format_str.format(fmt(self.format_time.min()), fmt(self.format_time.max()), fmt(self.format_time.avg()), fmt(self.format_time.get()))
            if self.collate_time.get():
                out += '    * In collate_fn: {} min, {} max, {} avg, {} total\n'.format(fmt(self.collate_time.min()), fmt(self.collate_time.max()), fmt(self.collate_time.avg()), fmt(self.collate_time.get()))
            if self.finalize_batch_time.get():
                format_str = '   * In host->device transfer: {} min, {} max, {} avg, {} total\n'
                out += format_str.format(fmt(self.finalize_batch_time.min()), fmt(self.finalize_batch_time.max()), fmt(self.finalize_batch_time.avg()), fmt(self.finalize_batch_time.get()))
        return out

    def __repr__(self, level=0) -> str:
        indent = leveled_indent(level)
        return f'IterStatsSummary(\n{indent}   wait_time={fmt(self.wait_time.get()) or None},\n{indent}   get_time={fmt(self.get_time.get()) or None},\n{indent}   iter_blocks_local={self.iter_blocks_local or None},\n{indent}   iter_blocks_remote={self.iter_blocks_remote or None},\n{indent}   iter_unknown_location={self.iter_unknown_location or None},\n{indent}   next_time={fmt(self.next_time.get()) or None},\n{indent}   format_time={fmt(self.format_time.get()) or None},\n{indent}   user_time={fmt(self.user_time.get()) or None},\n{indent}   total_time={fmt(self.total_time.get()) or None},\n{indent})'