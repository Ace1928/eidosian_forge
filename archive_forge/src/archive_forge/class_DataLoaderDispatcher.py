import math
from contextlib import suppress
from typing import Callable, List, Optional, Union
import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler
from .logging import get_logger
from .state import AcceleratorState, DistributedType, GradientState, is_torch_xla_available
from .utils import (
class DataLoaderDispatcher(DataLoader, DataLoaderStateMixin):
    """
    Subclass of a PyTorch `DataLoader` that will iterate and preprocess on process 0 only, then dispatch on each
    process their part of the batch.

    Args:
        split_batches (`bool`, *optional*, defaults to `False`):
            Whether the resulting `DataLoader` should split the batches of the original data loader across devices or
            yield full batches (in which case it will yield batches starting at the `process_index`-th and advancing of
            `num_processes` batches at each iteration). Another way to see this is that the observed batch size will be
            the same as the initial `dataloader` if this option is set to `True`, the batch size of the initial
            `dataloader` multiplied by `num_processes` otherwise. Setting this option to `True` requires that the batch
            size of the `dataloader` is a round multiple of `batch_size`.
        skip_batches (`int`, *optional*, defaults to 0):
            The number of batches to skip at the beginning of an iteration.

    **Available attributes:**

        - **total_batch_size** (`int`) -- Total batch size of the dataloader across all processes.
            Equal to the original batch size when `split_batches=True`; otherwise the original batch size * the total
            number of processes

        - **total_dataset_length** (`int`) -- Total length of the inner dataset across all processes.
    """

    def __init__(self, dataset, split_batches: bool=False, skip_batches=0, _drop_last: bool=False, slice_fn=None, **kwargs):
        shuffle = False
        if is_torch_version('>=', '1.11.0'):
            from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
            if isinstance(dataset, ShufflerIterDataPipe):
                shuffle = dataset._shuffle_enabled
        super().__init__(dataset, **kwargs)
        self.split_batches = split_batches
        if shuffle:
            torch.utils.data.graph_settings.apply_shuffle_settings(dataset, shuffle=shuffle)
        self.gradient_state = GradientState()
        self.state = AcceleratorState()
        self._drop_last = _drop_last
        self.skip_batches = skip_batches
        self.slice_fn = slice_tensors if slice_fn is None else slice_fn
        self.iteration = 0

    def _fetch_batches(self, iterator):
        batches, batch = (None, None)
        if self.state.process_index == 0:
            try:
                if self.split_batches:
                    batch = next(iterator)
                else:
                    batches = []
                    for _ in range(self.state.num_processes):
                        batches.append(next(iterator))
                    try:
                        batch = concatenate(batches, dim=0)
                    except RuntimeError as e:
                        raise RuntimeError("You can't use batches of different size with `dispatch_batches=True` or when using an `IterableDataset`.either pass `dispatch_batches=False` and have each process fetch its own batch  or pass `split_batches=True`. By doing so, the main process will fetch a full batch and slice it into `num_processes` batches for each process.") from e
                batch_info = [get_data_structure(batch), False]
            except StopIteration:
                batch_info = [None, True]
        else:
            batch_info = [None, self._stop_iteration]
        broadcast_object_list(batch_info)
        self._stop_iteration = batch_info[1]
        if self._stop_iteration:
            if not self.split_batches and (not self._drop_last):
                if self.state.process_index == 0 and len(batches) > 0:
                    batch = concatenate(batches, dim=0)
                    batch_info = [get_data_structure(batch), False]
                else:
                    batch_info = [None, True]
                broadcast_object_list(batch_info)
        return (batch, batch_info)

    def __iter__(self):
        self.begin()
        self.set_epoch(self.iteration)
        main_iterator = None
        if is_torch_version('>=', '2.0.1'):
            main_iterator = super().__iter__()
        elif self.state.process_index == 0:
            main_iterator = super().__iter__()
        stop_iteration = False
        self._stop_iteration = False
        first_batch = None
        next_batch, next_batch_info = self._fetch_batches(main_iterator)
        batch_index = 0
        while not stop_iteration:
            batch, batch_info = (next_batch, next_batch_info)
            if self.state.process_index != 0:
                batch = initialize_tensors(batch_info[0])
            batch = send_to_device(batch, self.state.device)
            batch = broadcast(batch, from_process=0)
            if not self._drop_last and first_batch is None:
                first_batch = self.slice_fn(batch, slice(0, self.state.num_processes), process_index=self.state.process_index, num_processes=self.state.num_processes)
            if batch is None:
                raise ValueError(f'Batch does not contain any data (`{batch}`). At the end of all iterable data available before expected stop iteration.')
            observed_batch_size = find_batch_size(batch)
            batch_size = observed_batch_size // self.state.num_processes
            stop_iteration = self._stop_iteration
            if not stop_iteration:
                next_batch, next_batch_info = self._fetch_batches(main_iterator)
                if self._stop_iteration and next_batch_info[0] is None:
                    stop_iteration = True
            if not self._drop_last and stop_iteration and (observed_batch_size % self.state.num_processes != 0):
                batch = concatenate([batch, first_batch], dim=0)
                batch_size += 1
            data_slice = slice(self.state.process_index * batch_size, (self.state.process_index + 1) * batch_size)
            batch = self.slice_fn(batch, data_slice, process_index=self.state.process_index, num_processes=self.state.num_processes)
            if stop_iteration:
                self.end_of_dataloader = True
                self.remainder = observed_batch_size
            if batch_index >= self.skip_batches:
                yield batch
            batch_index += 1
        self.iteration += 1
        self.end()

    def set_epoch(self, epoch: int):
        if self.iteration != epoch:
            self.iteration = epoch
        if hasattr(self.batch_sampler.sampler, 'set_epoch'):
            self.batch_sampler.sampler.set_epoch(epoch)
        elif hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)

    def __len__(self):
        whole_length = super().__len__()
        if self.split_batches:
            return whole_length
        elif self._drop_last:
            return whole_length // self.state.num_processes
        else:
            return math.ceil(whole_length / self.state.num_processes)

    @property
    def total_batch_size(self):
        return self.dataset.batch_size if self.split_batches else self.dataset.batch_size * self.dataset.num_processes

    @property
    def total_dataset_length(self):
        return len(self.dataset)