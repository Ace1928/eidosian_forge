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
class LayerwiseMemoryTracker:
    """
    Observe a module to get the graph of the memory consumption during
    the forward and backward, layer by layer, with:
    - a breakdown of the memory used (activations memory estimation)
    - additional details such as amount of data exchanged with all gather

    Requires the model to be on a CUDA device to track its memory

    Example usage (no FSDP):

        ```
        # create your model
        model = models.resnet50().cuda()

        # monitor the model
        tracker = LayerwiseMemoryTracker()
        tracker.monitor(model)

        # Do a forward/backward
        criterion(model(input), target).backward()

        # show the plots
        tracker.show_plots()

        # get the detailed traces
        tracker.memory_traces

        # print a summary
        print(tracker.summary)
        ```

    Advanced usage (for FSDP):

        ```
        # wrap the group used for FSDP
        group = ProcessGroupTracker(group)

        # use this group when creating FSDP blocks
        model = FullyShardedDataParallel(model, process_group=group),

        # monitor the model as before
        tracker = LayerwiseMemoryTracker()
        tracker.monitor(model)

        # the detailed traces will now contain information
        # about the amount of all gathered data
        tracker.memory_traces
        ```
    """

    def __init__(self) -> None:
        self.memory_traces: List[LayerMemoryTrace] = []
        self._hooks: List[RemovableHandle] = []
        self._previous_module_name: Optional[str] = None
        self._last_all_gather_memory = 0
        self._cumul_all_gather_memory: List[int] = []
        self._memory_pre_forward = 0
        self._traced_module_names: Set[str] = set()

    def monitor(self, model: nn.Module) -> None:
        """
        Install hooks on the model to track its memory usage
        """
        for name, m in model.named_modules():
            h1 = m.register_forward_pre_hook(self._create_pre_forward_hook(name))
            h2 = m.register_forward_hook(self._create_post_forward_hook(name))
            h3 = m.register_backward_hook(self._create_backward_hook(name))
            self._hooks.extend([h1, h2, h3])
            if isinstance(m, FullyShardedDataParallel):
                if isinstance(m.process_group, ProcessGroupTracker):
                    m.process_group.listener = self._handle_process_group_call
        torch.cuda.empty_cache()

    def clear_traces(self) -> None:
        """
        Clear all the traces: new traces will be written on a clean slate
        """
        self.memory_traces.clear()

    def stop(self) -> None:
        """
        Stop any form of tracking (removes the hooks used to monitor the model)
        """
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._previous_module_name = None
        self._memory_pre_forward = 0
        self._last_all_gather_memory = 0
        self._cumul_all_gather_memory.clear()

    @property
    def forward_traces(self) -> List[LayerMemoryTrace]:
        """
        Get the part of the traces which corresponds to the forward pass
        """
        return [t for t in self.memory_traces if t.is_forward]

    @property
    def backward_traces(self) -> List[LayerMemoryTrace]:
        """
        Get the part of the traces which corresponds to the backward pass
        """
        return [t for t in self.memory_traces if not t.is_forward]

    @property
    def max_memory_allocated(self) -> int:
        """
        Peak memory allocated during the forward/backward pass
        """
        return max((t.allocated for t in self.memory_traces))

    @property
    def max_memory_cached(self) -> int:
        """
        Peak memory cached during the forward/backward pass
        """
        return max((t.reserved for t in self.memory_traces))

    @property
    def summary(self) -> LayerwiseMemoryTrackerSummary:
        """
        A quick summary of interesting statistics on the memory usage
        during the forward/backward pass
        """
        total_diff = sum((t.event.memory_diff for t in self.forward_traces))
        total_act = sum((t.event.memory_activations for t in self.forward_traces))
        top_act_producers = self.top_forward_activation_producers(top=10)
        return LayerwiseMemoryTrackerSummary(max_memory_allocated=self.max_memory_allocated, max_memory_cached=self.max_memory_cached, total_activation_allocations=total_act, total_forward_allocations=total_diff, top_forward_activation_producers=top_act_producers)

    def top_forward_activation_producers(self, top: int=10) -> List[LayerMemoryTrace]:
        """
        What are the top activation producers during the forward pass
        """
        return sorted(self.forward_traces, key=lambda a: a.event.memory_activations, reverse=True)[:top]

    def show_plots(self, figsize: Tuple[int, int]=(16, 20), capture: bool=False) -> Optional[Any]:
        """
        Show useful memory plots. Use "capture=True" to return an image
        rather than displaying the plots.
        """
        return compare_memory_traces_in_plot({'run': self.memory_traces}, figsize=figsize, capture=capture)

    def save_traces(self, path: str) -> None:
        """
        Save the traces in a JSON file
        """
        import json
        with open(path, 'w') as f:
            json_traces = [t.to_dict() for t in self.memory_traces]
            json.dump({'traces': json_traces}, f)

    @classmethod
    def load(cls, path: str) -> 'LayerwiseMemoryTracker':
        import json
        out = cls()
        with open(path, 'r') as f:
            traces = json.load(f)['traces']
        out.memory_traces = [LayerMemoryTrace.from_dict(t) for t in traces]
        return out

    def _create_pre_forward_hook(self, name: str) -> Callable:

        def _pre_forward_hook(module: nn.Module, inputs: Any) -> None:
            torch.cuda.synchronize()
            allocated, reserved = self._capture_memory()
            self._previous_module_name = name
            self._memory_pre_forward = allocated
            if isinstance(module, FullyShardedDataParallel):
                self._cumul_all_gather_memory.append(0)
        return _pre_forward_hook

    def _handle_process_group_call(self, event: ProcessGroupTrackingEvent, *args: Sequence[Any]) -> None:
        torch.cuda.synchronize()
        if event == ProcessGroupTrackingEvent.allgather:
            outputs, inputs = args
            output_size = self._get_module_output_size(outputs)
            self._last_all_gather_memory += output_size
            if self._cumul_all_gather_memory:
                self._cumul_all_gather_memory[-1] += output_size

    def _create_post_forward_hook(self, name: str) -> Callable:

        def _post_forward_hook(module: nn.Module, inputs: Sequence[torch.Tensor], outputs: Sequence[torch.Tensor]) -> None:
            torch.cuda.synchronize()
            if isinstance(module, FullyShardedDataParallel):
                self._cumul_all_gather_memory.pop()
            if name == self._previous_module_name:
                allocated, reserved = self._capture_memory()
                self._traced_module_names.add(name)
                ys = self._filter_allocated_output(inputs, outputs)
                activations = sum((self._get_module_output_size(y) for y in ys))
                self.memory_traces.append(LayerMemoryTrace(module_name=name, module_params=self._get_parameter_size(module), allocated=allocated, reserved=reserved, is_forward=True, all_gathered=self._last_all_gather_memory, cumul_all_gathered=sum(self._cumul_all_gather_memory), event=TraceForwardEvent(memory_diff=allocated - self._memory_pre_forward, memory_activations=activations)))
                self._last_all_gather_memory = 0
            self._previous_module_name = None
            self._memory_pre_forward = 0
        return _post_forward_hook

    def _create_backward_hook(self, name: str) -> Callable:

        def _backward_hook(module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor) -> None:
            torch.cuda.synchronize()
            if name not in self._traced_module_names:
                return
            ys = self._filter_allocated_output(grad_input, grad_output)
            memory = sum((self._get_module_output_size(y) for y in ys))
            allocated, reserved = self._capture_memory()
            self.memory_traces.append(LayerMemoryTrace(module_name=name, module_params=self._get_parameter_size(module), allocated=allocated, reserved=reserved, is_forward=False, all_gathered=self._last_all_gather_memory, cumul_all_gathered=0, event=TraceBackwardEvent(memory_activations=memory)))
            self._last_all_gather_memory = 0
        return _backward_hook

    @staticmethod
    def _capture_memory() -> Tuple[int, int]:
        torch.cuda.synchronize()
        allocated_mb = torch.cuda.memory_allocated()
        reserved_mb = torch.cuda.memory_reserved()
        return (allocated_mb, reserved_mb)

    @classmethod
    def _get_parameter_size(cls, module: nn.Module) -> int:
        return sum((p.numel() * cls._get_dtype_size(p) for p in module.parameters()))

    @classmethod
    def _get_module_output_size(cls, xs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> int:
        """
        Return the minimum memory requirement to store the tensors
        provided as parameters
        """
        if isinstance(xs, torch.Tensor):
            x = xs
            p = cls._get_dtype_size(x)
            for d in x.shape:
                p *= d
            return p
        elif isinstance(xs, tuple) or isinstance(xs, list):
            return sum((cls._get_module_output_size(x) for x in xs))
        return 0

    @classmethod
    def _get_dtype_size(cls, x: torch.Tensor) -> int:
        return 2 if x.dtype == torch.float16 else 4

    @classmethod
    def _filter_allocated_output(cls, inputs: Union[torch.Tensor, Sequence[torch.Tensor]], outputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> List[torch.Tensor]:
        """
        Only return the outputs that are allocated and not views, reshape
        or stride of the inputs
        """
        xs = cls._collect_tensors(inputs)
        ys = cls._collect_tensors(outputs)
        return [y for y in ys if all((not cls._is_same_storage(x, y) for x in xs))]

    @staticmethod
    def _is_same_storage(x: torch.Tensor, y: torch.Tensor) -> bool:
        """
        Indicate if x and y share the same storage, meaning that one of them
        is a view, reshape or stride of the other or from a common tensor
        """
        return x.storage().data_ptr() == y.storage().data_ptr()

    @staticmethod
    def _collect_tensors(module_io_tensors: Union[torch.Tensor, Sequence[torch.Tensor]]) -> List[torch.Tensor]:
        """
        Extract the tensors out of the provided input or output of a nn.Module
        """
        tensors = []
        to_visit = [module_io_tensors]
        while to_visit:
            x = to_visit.pop()
            if isinstance(x, torch.Tensor):
                tensors.append(x)
            elif isinstance(x, tuple) or isinstance(x, list):
                to_visit.extend(module_io_tensors)
        return tensors