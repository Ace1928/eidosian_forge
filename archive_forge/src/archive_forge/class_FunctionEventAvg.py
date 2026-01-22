import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
class FunctionEventAvg(FormattedTimesMixin):
    """Used to average stats over multiple FunctionEvent objects."""

    def __init__(self):
        self.key: Optional[str] = None
        self.count: int = 0
        self.node_id: int = 0
        self.is_async: bool = False
        self.is_remote: bool = False
        self.use_device: Optional[str] = None
        self.cpu_time_total: int = 0
        self.cuda_time_total: int = 0
        self.privateuse1_time_total: int = 0
        self.self_cpu_time_total: int = 0
        self.self_cuda_time_total: int = 0
        self.self_privateuse1_time_total: int = 0
        self.input_shapes: Optional[List[List[int]]] = None
        self.stack: Optional[List] = None
        self.scope: Optional[int] = None
        self.cpu_memory_usage: int = 0
        self.cuda_memory_usage: int = 0
        self.privateuse1_memory_usage: int = 0
        self.self_cpu_memory_usage: int = 0
        self.self_cuda_memory_usage: int = 0
        self.self_privateuse1_memory_usage: int = 0
        self.cpu_children: Optional[List[FunctionEvent]] = None
        self.cpu_parent: Optional[FunctionEvent] = None
        self.device_type: DeviceType = DeviceType.CPU
        self.is_legacy: bool = False
        self.flops: int = 0

    def add(self, other):
        if self.key is None:
            self.key = other.key
            self.node_id = other.node_id
            self.is_async = other.is_async
            self.is_remote = other.is_remote
            self.cpu_parent = other.cpu_parent
            self.cpu_children = other.cpu_children
            self.input_shapes = other.input_shapes
            self.stack = other.stack
            self.scope = other.scope
            self.device_type = other.device_type
            self.is_legacy = other.is_legacy
            self.use_device = other.use_device
        assert isinstance(other, (FunctionEvent, FunctionEventAvg))
        assert other.key == self.key
        self.cpu_time_total += other.cpu_time_total
        self.cuda_time_total += other.cuda_time_total
        self.privateuse1_time_total += other.privateuse1_time_total
        self.self_cpu_time_total += other.self_cpu_time_total
        self.self_cuda_time_total += other.self_cuda_time_total
        self.self_privateuse1_time_total += other.self_privateuse1_time_total
        self.cpu_memory_usage += other.cpu_memory_usage
        self.cuda_memory_usage += other.cuda_memory_usage
        self.privateuse1_memory_usage += other.privateuse1_memory_usage
        self.self_cpu_memory_usage += other.self_cpu_memory_usage
        self.self_cuda_memory_usage += other.self_cuda_memory_usage
        self.self_privateuse1_memory_usage += other.self_privateuse1_memory_usage
        self.count += other.count
        if self.flops is None:
            self.flops = other.flops
        elif other.flops is not None:
            self.flops += other.flops
        return self

    def __iadd__(self, other):
        return self.add(other)

    def __repr__(self):
        device_name = 'cuda' if not self.use_device else self.use_device
        self_device_time = self.self_cuda_time_total_str if not self.use_device else self.self_privateuse1_time_total_str
        device_time = self.cuda_time_str if not self.use_device else self.privateuse1_time_str
        device_memory = self.cuda_memory_usage if not self.use_device else self.privateuse1_memory_usage
        return '<FunctionEventAvg key={} self_cpu_time={} cpu_time={}  self_{}_time={} {}_time={} input_shapes={} cpu_memory_usage={} {}_memory_usage={}>'.format(self.key, self.self_cpu_time_total_str, self.cpu_time_str, device_name, self_device_time, device_name, device_time, str(self.input_shapes), self.cpu_memory_usage, device_name, device_memory)