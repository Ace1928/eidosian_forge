import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from ..utils.generic import ModelOutput
class PipelinePackIterator(PipelineIterator):
    """
    Roughly equivalent to

    ```
    packed =  []
    for item in loader:
        packed.append(item)
        if item["is_last"]:
            yield packed
            packed = []
    ```

        but it also handles cases where `item` are batched (meaning it's a dict of Tensor with first dimension > 1. In
        that case it does

    ```
    packed =  []
    for batch in loader:
        # item is batched
        for item in batch:
            packed.append(item)
            if item["is_last"]:
                yield packed
                packed = []
    ```

        Arguments:
            loader (`torch.utils.data.DataLoader` or any iterator):
                The iterator that will be used to apply `infer` on.
            infer (any function):
                The function to apply of each element of `loader`.
            params (`dict`):
                The parameters passed to `infer` along with every item
            loader_batch_size (`int`, *optional*):
                If specified, the items of `loader` are supposed to come as batch, and are loader_batched here making
                it roughly behave as


    ```
    for items in loader:
        for i in loader_batch_size:
            item = items[i]
            yield infer(item, **params)
    ```"""

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

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