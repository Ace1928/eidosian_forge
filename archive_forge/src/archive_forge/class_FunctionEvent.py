import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
class FunctionEvent(FormattedTimesMixin):
    """Profiling information about a single function."""

    def __init__(self, id, name, thread, start_us, end_us, fwd_thread=None, input_shapes=None, stack=None, scope=0, use_device=None, cpu_memory_usage=0, cuda_memory_usage=0, privateuse1_memory_usage=0, is_async=False, is_remote=False, sequence_nr=-1, node_id=-1, device_type=DeviceType.CPU, device_index=0, is_legacy=False, flops=None, trace_name=None, concrete_inputs=None):
        self.id: int = id
        self.node_id: int = node_id
        self.name: str = name
        self.trace_name: str = trace_name
        self.time_range: Interval = Interval(start_us, end_us)
        self.thread: int = thread
        self.fwd_thread: Optional[int] = fwd_thread
        self.kernels: List[Kernel] = []
        self.count: int = 1
        self.cpu_children: List[FunctionEvent] = []
        self.cpu_parent: Optional[FunctionEvent] = None
        self.input_shapes: Tuple[int, ...] = input_shapes
        self.concrete_inputs: List[Any] = concrete_inputs
        self.stack: List = stack
        self.scope: int = scope
        self.use_device: Optional[str] = use_device
        self.cpu_memory_usage: int = cpu_memory_usage
        self.cuda_memory_usage: int = cuda_memory_usage
        self.privateuse1_memory_usage: int = privateuse1_memory_usage
        self.is_async: bool = is_async
        self.is_remote: bool = is_remote
        self.sequence_nr: int = sequence_nr
        self.device_type: DeviceType = device_type
        self.device_index: int = device_index
        self.is_legacy: bool = is_legacy
        self.flops: Optional[int] = flops

    def append_kernel(self, name, device, duration):
        assert self.device_type == DeviceType.CPU
        self.kernels.append(Kernel(name, device, duration))

    def append_cpu_child(self, child):
        """Append a CPU child of type FunctionEvent.

        One is supposed to append only direct children to the event to have
        correct self cpu time being reported.
        """
        assert self.device_type == DeviceType.CPU
        assert isinstance(child, FunctionEvent)
        assert child.device_type == DeviceType.CPU
        self.cpu_children.append(child)

    def set_cpu_parent(self, parent):
        """Set the immediate CPU parent of type FunctionEvent.

        One profiling FunctionEvent should have only one CPU parent such that
        the child's range interval is completely inside the parent's. We use
        this connection to determine the event is from top-level op or not.
        """
        assert self.device_type == DeviceType.CPU
        assert isinstance(parent, FunctionEvent)
        assert parent.device_type == DeviceType.CPU
        self.cpu_parent = parent

    @property
    def self_cpu_memory_usage(self):
        if self.is_async or self.device_type != DeviceType.CPU:
            return 0
        return self.cpu_memory_usage - sum([child.cpu_memory_usage for child in self.cpu_children])

    @property
    def self_cuda_memory_usage(self):
        if self.is_async or self.device_type != DeviceType.CPU:
            return 0
        return self.cuda_memory_usage - sum([child.cuda_memory_usage for child in self.cpu_children])

    @property
    def self_privateuse1_memory_usage(self):
        if self.is_async or self.device_type != DeviceType.CPU:
            return 0
        return self.privateuse1_memory_usage - sum([child.privateuse1_memory_usage for child in self.cpu_children])

    @property
    def self_cpu_time_total(self):
        if self.is_async or self.device_type != DeviceType.CPU:
            return 0
        return self.cpu_time_total - sum([child.cpu_time_total for child in self.cpu_children])

    @property
    def cuda_time_total(self):
        if self.is_async or self.use_device:
            return 0
        if self.device_type == DeviceType.CPU:
            if not self.is_legacy:
                return sum((kinfo.duration for kinfo in self.kernels)) + sum((ch.cuda_time_total for ch in self.cpu_children))
            else:
                return sum((kinfo.duration for kinfo in self.kernels))
        else:
            assert self.device_type == DeviceType.CUDA
            return self.time_range.elapsed_us()

    @property
    def self_cuda_time_total(self):
        if self.is_async or self.use_device:
            return 0
        if self.device_type == DeviceType.CPU:
            return self.cuda_time_total - sum([child.cuda_time_total for child in self.cpu_children])
        else:
            assert self.device_type == DeviceType.CUDA
            return self.cuda_time_total

    @property
    def cpu_time_total(self):
        if self.device_type == DeviceType.CPU:
            return self.time_range.elapsed_us()
        else:
            return 0

    @property
    def self_privateuse1_time_total(self):
        if self.is_async or not self.use_device:
            return 0
        if self.device_type == DeviceType.CPU:
            return self.privateuse1_time_total - sum([child.privateuse1_time_total for child in self.cpu_children])
        else:
            assert self.device_type == DeviceType.CUDA
            return self.privateuse1_time_total

    @property
    def privateuse1_time_total(self):
        if self.is_async or not self.use_device:
            return 0
        if self.device_type == DeviceType.CPU:
            if not self.is_legacy:
                return sum((kinfo.duration for kinfo in self.kernels)) + sum((ch.privateuse1_time_total for ch in self.cpu_children))
            else:
                return sum((kinfo.duration for kinfo in self.kernels))
        else:
            assert self.device_type == DeviceType.PrivateUse1
            return self.time_range.elapsed_us()

    @property
    def key(self):
        return self.name

    def __repr__(self):
        device_name = 'cuda' if not self.use_device else self.use_device
        device_time = self.cuda_time_str if not self.use_device else self.privateuse1_time_str
        device_memory_usage = self.cuda_memory_usage if not self.use_device else self.privateuse1_memory_usage
        return '<FunctionEvent id={} name={} device_type={} node_id={} cpu_time={} start_us={} end_us={} cpu_children={} {}_time={} name={} thread={} input_shapes={} cpu_memory_usage={} {}_memory_usage={} is_async={} is_remote={} seq_nr={} is_legacy={}>'.format(self.id, self.name, self.device_type, self.node_id, self.cpu_time_str, self.time_range.start, self.time_range.end, str([child.id for child in self.cpu_children]), device_name, device_time, self.name, self.thread, str(self.input_shapes), self.cpu_memory_usage, device_name, device_memory_usage, self.is_async, self.is_remote, self.sequence_nr, self.is_legacy)