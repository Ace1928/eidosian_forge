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
class _MemoryGraphCreator:
    """
    Helper class to create graphs to display memory
    """

    def __init__(self) -> None:
        import matplotlib
        self.font = {'family': matplotlib.rcParams['font.family'], 'weight': 'normal', 'size': 12}

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

    def reserved_memory_curve(self, ax: Any, job_name: str, memory_traces: List[LayerMemoryTrace]) -> None:
        reserved_memory = [t.reserved for t in memory_traces]
        x, y_forward, y_backward = self._split_forward_backward(memory_traces, reserved_memory)
        ax.plot(x, y_forward, x, y_backward, label=job_name)
        self._y_axis_in_gigabytes(ax)

    def activation_allocations(self, ax: Any, job_name: str, memory_traces: List[LayerMemoryTrace]) -> None:
        event_allocations = [t.event.memory_activations for t in memory_traces]
        x, y_forward, y_backward = self._split_forward_backward(memory_traces, event_allocations)
        ax.plot(x, y_forward, x, y_backward, label=job_name)
        self._y_axis_in_gigabytes(ax)

    def cumulative_activations(self, ax: Any, job_name: str, memory_traces: List[LayerMemoryTrace]) -> None:
        event_allocations = [t.event.memory_activations for t in memory_traces]
        x, y_forward, y_backward = self._split_forward_backward(memory_traces, event_allocations)
        cumulative_forward_activations = np.cumsum(y_forward)
        ax.plot(x, cumulative_forward_activations, label=job_name)
        self._y_axis_in_gigabytes(ax)

    def all_gathered_memory(self, ax: Any, job_name: str, memory_traces: List[LayerMemoryTrace]) -> None:
        gathered_memory = [t.all_gathered for t in memory_traces]
        cumul_gathered_memory = [t.cumul_all_gathered for t in memory_traces]
        x, y_forward, y_backward = self._split_forward_backward(memory_traces, gathered_memory)
        ax.plot(x, y_forward, x, y_backward, label=job_name)
        ax.plot(x, cumul_gathered_memory, label=job_name)
        self._y_axis_in_gigabytes(ax)
        max_index = np.argmax(cumul_gathered_memory)
        max_trace = memory_traces[max_index]
        max_module = '.'.join([n for n in max_trace.module_name.split('.') if not n.startswith('_')])
        ax.set_ylim([None, max_trace.cumul_all_gathered * 1.1])
        x_text, y_text = (max(0, max_index * 0.8), max_trace.cumul_all_gathered * 1.04)
        ax.text(x_text, y_text, f'{max_module} (fwd)', fontdict=self.font)

    def module_parameters(self, ax: Any, job_name: str, memory_traces: List[LayerMemoryTrace]) -> None:
        module_parameters = [t.module_params for t in memory_traces]
        x, y_forward, y_backward = self._split_forward_backward(memory_traces, module_parameters)
        ax.plot(x, y_forward, x, y_backward, label=job_name)
        self._y_axis_in_gigabytes(ax)

    @staticmethod
    def _y_axis_in_gigabytes(ax: Any) -> None:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(9, 9))

    @classmethod
    def _split_forward_backward(cls, memory_traces: List[LayerMemoryTrace], values: List[Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_values = np.array(list(range(len(memory_traces))))
        mask_forwards, mask_backwards = cls._mask_forward_backward(memory_traces)
        return (x_values, np.ma.masked_where(mask_backwards, values), np.ma.masked_where(mask_forwards, values))

    @classmethod
    def _mask_forward_backward(cls, memory_traces: List[LayerMemoryTrace]) -> Tuple[np.ndarray, np.ndarray]:
        mask_forwards = np.array([t.is_forward for t in memory_traces])
        return (mask_forwards, ~mask_forwards)