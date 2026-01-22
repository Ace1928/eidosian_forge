import warnings
from abc import ABC, abstractmethod
from collections import deque
import copy as copymodule
from typing import Any, Callable, Iterator, List, Literal, Optional, Sized, Tuple, TypeVar, Deque
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper, _check_unpickable_fn
class _ForkerIterDataPipe(IterDataPipe, _ContainerTemplate):
    """
    Container to hold instance-specific information on behalf of ForkerIterDataPipe.

    It tracks the state of its child DataPipes, maintains the buffer, and yields the next value
    as requested by the child DataPipes.
    """

    def __init__(self, datapipe: IterDataPipe, num_instances: int, buffer_size: int=1000, copy: Optional[Literal['shallow', 'deep']]=None):
        self.main_datapipe = datapipe
        self._datapipe_iterator: Optional[Iterator[Any]] = None
        self.num_instances = num_instances
        self.buffer: Deque = deque()
        self.buffer_size = buffer_size
        if self.buffer_size < 0:
            warnings.warn('Unlimited buffer size is set for `fork`, please be aware of OOM at random places', UserWarning)
        if copy is None:
            self.copy_fn = _no_op
        elif copy == 'shallow':
            self.copy_fn = copymodule.copy
        elif copy == 'deep':
            self.copy_fn = copymodule.deepcopy
        else:
            raise ValueError(f'Unknown copy method `{copy}` requested, choose one of None, `shallow` or `deep`.')
        self.child_pointers: List[int] = [0] * num_instances
        self.slowest_ptr = 0
        self.leading_ptr = 0
        self.end_ptr: Optional[int] = None
        self._child_stop: List[bool] = [True for _ in range(num_instances)]

    def __len__(self):
        return len(self.main_datapipe)

    def get_next_element_by_instance(self, instance_id: int):
        if self._datapipe_iterator is None and self._child_stop[instance_id]:
            self._datapipe_iterator = iter(self.main_datapipe)
            self._snapshot_state = _SnapshotState.Iterating
            for i in range(self.num_instances):
                self._child_stop[i] = False
        try:
            while not self._child_stop[instance_id]:
                self.child_pointers[instance_id] += 1
                if self.end_ptr is not None and self.child_pointers[instance_id] == self.end_ptr:
                    self._child_stop[instance_id] = True
                    break
                if self.buffer and self.child_pointers[instance_id] <= self.leading_ptr:
                    idx = self.child_pointers[instance_id] - self.slowest_ptr - 1
                    return_val = self.buffer[idx]
                else:
                    self.leading_ptr = self.child_pointers[instance_id]
                    try:
                        return_val = next(self._datapipe_iterator)
                        self.buffer.append(return_val)
                    except StopIteration:
                        self._child_stop[instance_id] = True
                        self._datapipe_iterator = None
                        self.end_ptr = self.leading_ptr
                        continue
                if self.child_pointers[instance_id] == self.slowest_ptr + 1:
                    new_min = min(self.child_pointers)
                    if self.slowest_ptr < new_min:
                        self.slowest_ptr = new_min
                        self.buffer.popleft()
                if self.buffer_size >= 0 and self.leading_ptr > self.buffer_size + self.slowest_ptr:
                    raise BufferError('ForkerIterDataPipe buffer overflow,' + f'buffer size {self.buffer_size} is insufficient.')
                yield self.copy_fn(return_val)
        finally:
            self._child_stop[instance_id] = True
            if all(self._child_stop):
                self._datapipe_iterator = None
                self._cleanup()

    def is_every_instance_exhausted(self) -> bool:
        return self.end_ptr is not None and all(self._child_stop)

    def get_length_by_instance(self, instance_id: int) -> int:
        return len(self.main_datapipe)

    def reset(self) -> None:
        self._datapipe_iterator = None
        self.buffer = deque()
        self.child_pointers = [0] * self.num_instances
        self.slowest_ptr = 0
        self.leading_ptr = 0
        self.end_ptr = None
        self._child_stop = [True for _ in range(self.num_instances)]

    def __getstate__(self):
        state = (self.main_datapipe, self.num_instances, self.buffer_size, self.copy_fn, self._valid_iterator_id, self._number_of_samples_yielded)
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        self.main_datapipe, self.num_instances, self.buffer_size, self.copy_fn, self._valid_iterator_id, self._number_of_samples_yielded = state
        self._datapipe_iterator = None
        self.buffer = deque()
        self.child_pointers = [0] * self.num_instances
        self.slowest_ptr = 0
        self.leading_ptr = 0
        self.end_ptr = None
        self._child_stop = [True for _ in range(self.num_instances)]

    def _cleanup(self):
        while self.buffer:
            d = self.buffer.popleft()
            StreamWrapper.close_streams(d)

    def __del__(self):
        self._cleanup()