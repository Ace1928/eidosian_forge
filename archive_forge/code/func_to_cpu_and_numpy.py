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
from .utils import (
def to_cpu_and_numpy(self) -> None:
    """Move tensors in stored objects to CPU and convert them to numpy arrays."""
    if self.tensors is None:
        return
    new_arrays = nested_numpify(self.tensors)
    if self.arrays is None:
        self.arrays = new_arrays
    elif self.do_nested_concat:
        self.arrays = nested_concat(self.arrays, new_arrays, padding_index=self.padding_index)
    else:
        self.arrays.extend(new_arrays)
    self.tensors = None