import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from ..utils.generic import ModelOutput
def __next__(self):
    is_last = False
    accumulator = []
    if self._loader_batch_index is not None and self._loader_batch_index < self.loader_batch_size:
        while self._loader_batch_index < self.loader_batch_size:
            item = self.loader_batch_item()
            is_last = item.pop('is_last')
            accumulator.append(item)
            if is_last:
                return accumulator
    while not is_last:
        processed = self.infer(next(self.iterator), **self.params)
        if self.loader_batch_size is not None:
            if isinstance(processed, torch.Tensor):
                first_tensor = processed
            else:
                key = list(processed.keys())[0]
                first_tensor = processed[key]
            if isinstance(first_tensor, list):
                observed_batch_size = len(first_tensor)
            else:
                observed_batch_size = first_tensor.shape[0]
            if 0 < observed_batch_size < self.loader_batch_size:
                self.loader_batch_size = observed_batch_size
            self._loader_batch_data = processed
            self._loader_batch_index = 0
            while self._loader_batch_index < self.loader_batch_size:
                item = self.loader_batch_item()
                is_last = item.pop('is_last')
                accumulator.append(item)
                if is_last:
                    return accumulator
        else:
            item = processed
            is_last = item.pop('is_last')
            accumulator.append(item)
    return accumulator