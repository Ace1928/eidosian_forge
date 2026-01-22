from abc import ABC, abstractmethod
import queue
import threading
import collections
from dataclasses import dataclass
import os
import dataclasses
import io
import pickle
from typing import List, Union, Dict, cast
import torch
from torch import Tensor
from torch.futures import Future
from pathlib import Path
from .metadata import (
from .storage import (
from .planner import (
from .utils import _create_file_view
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch._utils import _get_device_module
class _OverlappingCpuLoader(_TensorLoader):

    def __init__(self, resolve_fun, stream=None, inflight_threshhold=1000000):
        self.resolve_fun = resolve_fun
        self.items = []
        self.inflight_threshhold = inflight_threshhold
        self.in_flight_data = 0
        self.current_items: collections.deque = collections.deque()
        self.idx = 0
        self.started = False
        self.device_type = stream.device_type if stream else torch.device('cuda').type
        self.device_module = _get_device_module(self.device_type)
        self.stream = stream or self.device_module.current_stream()
        if self.stream != self.device_module.current_stream():
            self.stream.wait_stream(self.device_module.current_stream())

    @property
    def _done(self):
        return self.idx >= len(self.items)

    def _drain(self):
        drained = []
        if self.in_flight_data >= self.inflight_threshhold:
            self.stream.synchronize()
        while self.in_flight_data >= self.inflight_threshhold:
            val = self.current_items.popleft()
            self.in_flight_data -= val[0].numel() * val[0].element_size()
            drained.append(val)
        return drained

    def _refill(self):
        with self.device_module.stream(self.stream):
            while not self._done and self.in_flight_data < self.inflight_threshhold:
                _, obj = self.items[self.idx]
                self.idx += 1
                tensor = self.resolve_fun(obj).detach()
                if tensor.device.type == self.device_type:
                    tensor = tensor.to(device='cpu', non_blocking=True)
                elif tensor.device == torch.device('cpu'):
                    if tensor.storage().size() != tensor.numel():
                        tensor = tensor.clone()
                self.current_items.append((tensor, obj))
                self.in_flight_data += tensor.numel() * tensor.element_size()

    def _finish(self):
        assert self._done
        if len(self.current_items) > 0:
            self.stream.synchronize()
        return self.current_items

    def add(self, size, obj):
        if self.started:
            raise RuntimeError('cannot add items after loading started')
        self.items.append((size, obj))

    def start_loading(self):
        if self.started:
            return
        self.started = True
        self.items.sort(key=lambda x: x[0])
        self._refill()

    def values(self):
        self.start_loading()
        while not self._done:
            drained = self._drain()
            self._refill()
            yield from drained
        yield from self._finish()