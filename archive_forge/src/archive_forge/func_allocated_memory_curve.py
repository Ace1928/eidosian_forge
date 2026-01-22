from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from fairscale.nn import FullyShardedDataParallel
def allocated_memory_curve(self, ax: Any, job_name: str, memory_traces: List[LayerMemoryTrace]) -> None:
    allocated_memory = [t.allocated for t in memory_traces]
    x, y_forward, y_backward = self._split_forward_backward(memory_traces, allocated_memory)
    ax.plot(x, y_forward, x, y_backward, label=job_name)
    max_index = np.argmax(allocated_memory)
    max_trace = memory_traces[max_index]
    max_module = '.'.join([n for n in max_trace.module_name.split('.') if not n.startswith('_')])
    max_phase = 'fwd' if max_trace.is_forward else 'bwd'
    ax.set_ylim([None, max_trace.allocated * 1.1])
    x_text, y_text = (max(0, max_index * 0.8), max_trace.allocated * 1.04)
    ax.text(x_text, y_text, f'{max_module} ({max_phase})', fontdict=self.font)
    self._y_axis_in_gigabytes(ax)