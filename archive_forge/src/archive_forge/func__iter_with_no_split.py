import math
from contextlib import suppress
from typing import Callable, List, Optional, Union
import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler
from .logging import get_logger
from .state import AcceleratorState, DistributedType, GradientState, is_torch_xla_available
from .utils import (
def _iter_with_no_split(self):
    initial_data = []
    batch_to_yield = []
    for idx, batch in enumerate(self.batch_sampler):
        if not self.drop_last and idx < self.num_processes:
            initial_data += batch
        if idx % self.num_processes == self.process_index:
            batch_to_yield = batch
        if idx % self.num_processes == self.num_processes - 1 and (self.batch_size is None or len(batch) == self.batch_size):
            yield batch_to_yield
            batch_to_yield = []
    if not self.drop_last and len(initial_data) > 0:
        if not self.even_batches:
            if len(batch_to_yield) > 0:
                yield batch_to_yield
        else:
            if len(batch_to_yield) == self.batch_size:
                yield batch_to_yield
            while len(initial_data) < self.num_processes * self.batch_size:
                initial_data += initial_data
            if len(batch) == self.batch_size:
                batch = []
                idx += 1
            cycle_index = 0
            while idx % self.num_processes != 0 or len(batch) > 0:
                end_index = cycle_index + self.batch_size - len(batch)
                batch += initial_data[cycle_index:end_index]
                if idx % self.num_processes == self.process_index:
                    yield batch
                cycle_index = end_index
                batch = []
                idx += 1