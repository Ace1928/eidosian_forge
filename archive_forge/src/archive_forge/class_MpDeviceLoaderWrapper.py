import math
from contextlib import suppress
from typing import Callable, List, Optional, Union
import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler
from .logging import get_logger
from .state import AcceleratorState, DistributedType, GradientState, is_torch_xla_available
from .utils import (
class MpDeviceLoaderWrapper(xpl.MpDeviceLoader):
    """
        Wrapper for the xpl.MpDeviceLoader class that knows the total batch size.

        XLA preloading threads will all call DataLoaderShard's __iter__(). Remove rng_types from DataLoaderShard to
        prevent it from using the XLA device in the preloading threads, and synchronize the RNG once from the main
        thread only.

        **Available attributes:**

        - **total_batch_size** (`int`) -- Total batch size of the dataloader across all processes.
            Equal to the original batch size when `split_batches=True`; otherwise the original batch size * the total
            number of processes

        - **total_dataset_length** (`int`) -- Total length of the inner dataset across all processes.
        """

    def __init__(self, dataloader: DataLoaderShard, device: torch.device):
        super().__init__(dataloader, device)
        self._rng_types = self._loader.rng_types
        self._loader.rng_types = None

    def __iter__(self):
        if self._rng_types is not None:
            synchronize_rng_states(self._rng_types, self._loader.synchronized_generator)
        return super().__iter__()

    @property
    def total_batch_size(self):
        return self._loader.total_batch_size

    @property
    def total_dataset_length(self):
        return self._loader.total_dataset_length

    @property
    def batch_sampler(self):
        return self._loader.batch_sampler