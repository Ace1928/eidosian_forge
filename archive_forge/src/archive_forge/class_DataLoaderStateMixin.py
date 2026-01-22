import math
from contextlib import suppress
from typing import Callable, List, Optional, Union
import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler
from .logging import get_logger
from .state import AcceleratorState, DistributedType, GradientState, is_torch_xla_available
from .utils import (
class DataLoaderStateMixin:
    """
    Mixin class that adds a state to a `DataLoader` to keep track of the status inside the dataloader such as at the
    end of the iteration, the number of items in the dataset in the last batch relative to the batch size, and other
    useful information that might be needed.

    **Available attributes:**

        - **end_of_dataloader** (`bool`) -- Whether at the last iteration or batch
        - **remainder** (`int`) -- The number of items that are remaining in the last batch, relative to the total
          batch size

    """

    def __init_subclass__(cls, **kwargs):
        cls.end_of_dataloader = False
        cls.remainder = -1

    def reset(self):
        self.end_of_dataloader = False
        self.remainder = -1

    def begin(self):
        """Prepares the gradient state for the current dataloader"""
        self.reset()
        with suppress(Exception):
            if not self._drop_last:
                length = getattr(self.dataset, 'total_dataset_length', len(self.dataset))
                self.remainder = length % self.total_batch_size
        self.gradient_state._add_dataloader(self)

    def end(self):
        """Cleans up the gradient state after exiting the dataloader"""
        self.gradient_state._remove_dataloader(self)