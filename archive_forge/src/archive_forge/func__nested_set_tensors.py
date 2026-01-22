import copy
import datetime
import io
import json
import math
import os
import sys
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import StreamHandler
from typing import Any, Dict, Iterator, List, Optional, Union
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, IterableDataset, RandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
from .integrations.deepspeed import is_deepspeed_zero3_enabled
from .tokenization_utils_base import BatchEncoding
from .utils import is_sagemaker_mp_enabled, is_torch_tpu_available, is_training_run_on_sagemaker, logging
def _nested_set_tensors(self, storage, arrays):
    if isinstance(arrays, (list, tuple)):
        result = [self._nested_set_tensors(x, y) for x, y in zip(storage, arrays)]
        return (result[0][0], type(arrays)((r[1] for r in result)))
    assert arrays.shape[0] % self.world_size == 0, f'Arrays passed should all have a first dimension multiple of {self.world_size}, found {arrays.shape[0]}.'
    slice_len = arrays.shape[0] // self.world_size
    for i in range(self.world_size):
        if len(arrays.shape) == 1:
            storage[self._offsets[i]:self._offsets[i] + slice_len] = arrays[i * slice_len:(i + 1) * slice_len]
        else:
            if len(storage.shape) > 1 and storage.shape[1] < arrays.shape[1]:
                storage = expand_like(storage, arrays.shape[1], padding_index=self.padding_index)
            storage[self._offsets[i]:self._offsets[i] + slice_len, :arrays.shape[1]] = arrays[i * slice_len:(i + 1) * slice_len]
    return (slice_len, storage)