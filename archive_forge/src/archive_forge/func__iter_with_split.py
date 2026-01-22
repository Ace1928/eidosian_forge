import math
from contextlib import suppress
from typing import Callable, List, Optional, Union
import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler
from .logging import get_logger
from .state import AcceleratorState, DistributedType, GradientState, is_torch_xla_available
from .utils import (
def _iter_with_split(self):
    initial_data = []
    batch_length = self.batch_sampler.batch_size // self.num_processes
    for idx, batch in enumerate(self.batch_sampler):
        if idx == 0:
            initial_data = batch
        if len(batch) == self.batch_size:
            yield batch[batch_length * self.process_index:batch_length * (self.process_index + 1)]
    if not self.drop_last and len(initial_data) > 0 and (len(batch) < self.batch_size):
        if not self.even_batches:
            if len(batch) > batch_length * self.process_index:
                yield batch[batch_length * self.process_index:batch_length * (self.process_index + 1)]
        else:
            while len(initial_data) < self.batch_size:
                initial_data += initial_data
            batch = batch + initial_data
            yield batch[batch_length * self.process_index:batch_length * (self.process_index + 1)]