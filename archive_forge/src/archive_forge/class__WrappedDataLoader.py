import collections
import logging
import os
import random
import types
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from packaging.version import Version
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import (
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.train._internal import session
from ray.train._internal.accelerator import Accelerator
from ray.train._internal.session import get_accelerator, set_accelerator
from ray.util.annotations import Deprecated, PublicAPI
class _WrappedDataLoader(DataLoader):

    def __init__(self, base_dataloader: DataLoader, device: torch.device, auto_transfer: bool):
        self.__dict__.update(getattr(base_dataloader, '__dict__', {}))
        self._dataloader = base_dataloader
        self.dataloader_iter = None
        self.device = device
        self._auto_transfer = auto_transfer if device.type == 'cuda' else False
        self._memcpy_stream = torch.cuda.Stream(device) if device.type == 'cuda' and self._auto_transfer else None
        self.next_batch = None

    def _move_to_device(self, item):
        if item is None:
            return None

        def try_move_device(i):
            try:
                i = i.to(self.device, non_blocking=self._auto_transfer)
            except AttributeError:
                logger.debug(f'Item {i} cannot be moved to device {self.device}.')
            return i
        with torch.cuda.stream(self._memcpy_stream):
            if isinstance(item, collections.abc.Mapping):
                item_on_device = {k: self._move_to_device(v) for k, v in item.items()}
            elif isinstance(item, tuple):
                item_on_device = tuple((self._move_to_device(i) for i in item))
            elif isinstance(item, list):
                item_on_device = [self._move_to_device(i) for i in item]
            elif isinstance(item, torch.Tensor):
                item_on_device = try_move_device(item)
            else:
                logger.debug(f"Data type {type(item)} doesn't support being moved to device.")
                item_on_device = item
            return item_on_device

    def _wait_for_batch(self, item):
        if self._memcpy_stream is None:
            return
        curr_stream = torch.cuda.current_stream()
        curr_stream.wait_stream(self._memcpy_stream)
        for i in item:
            try:
                i.record_stream(curr_stream)
            except AttributeError:
                pass

    def __len__(self):
        return len(self._dataloader)

    def _prefetch_next_batch(self):
        next_batch = next(self.dataloader_iter, None)
        self.next_batch = self._move_to_device(next_batch)

    def __iter__(self):
        self.dataloader_iter = iter(self._dataloader)
        self._prefetch_next_batch()
        return self

    def __next__(self):
        next_batch = self.next_batch
        if next_batch is None:
            raise StopIteration
        self._wait_for_batch(next_batch)
        self._prefetch_next_batch()
        return next_batch