import warnings
from abc import ABC, abstractmethod
from collections import deque
import copy as copymodule
from typing import Any, Callable, Iterator, List, Literal, Optional, Sized, Tuple, TypeVar, Deque
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper, _check_unpickable_fn
class _DemultiplexerIterDataPipe(IterDataPipe, _ContainerTemplate):
    """
    Container to hold instance-specific information on behalf of DemultiplexerIterDataPipe.

    It tracks the state of its child DataPipes, maintains the buffer, classifies and yields the next correct value
    as requested by the child DataPipes.
    """

    def __init__(self, datapipe: IterDataPipe[T_co], num_instances: int, classifier_fn: Callable[[T_co], Optional[int]], drop_none: bool, buffer_size: int):
        self.main_datapipe = datapipe
        self._datapipe_iterator: Optional[Iterator[Any]] = None
        self.num_instances = num_instances
        self.buffer_size = buffer_size
        if self.buffer_size < 0:
            warnings.warn('Unlimited buffer size is set for `demux`, please be aware of OOM at random places', UserWarning)
        self.current_buffer_usage = 0
        self.child_buffers: List[Deque[T_co]] = [deque() for _ in range(num_instances)]
        self.classifier_fn = classifier_fn
        self.drop_none = drop_none
        self.main_datapipe_exhausted = False
        self._child_stop: List[bool] = [True for _ in range(num_instances)]

    def _find_next(self, instance_id: int) -> T_co:
        while True:
            if self.main_datapipe_exhausted or self._child_stop[instance_id]:
                raise StopIteration
            if self._datapipe_iterator is None:
                raise ValueError('_datapipe_iterator has not been set, likely because this private method is called directly without invoking get_next_element_by_instance() first.')
            value = next(self._datapipe_iterator)
            classification = self.classifier_fn(value)
            if classification is None and self.drop_none:
                StreamWrapper.close_streams(value)
                continue
            if classification is None or classification >= self.num_instances or classification < 0:
                raise ValueError(f'Output of the classification fn should be between 0 and {self.num_instances - 1}. ' + f'{classification} is returned.')
            if classification == instance_id:
                return value
            self.child_buffers[classification].append(value)
            self.current_buffer_usage += 1
            if self.buffer_size >= 0 and self.current_buffer_usage > self.buffer_size:
                raise BufferError(f'DemultiplexerIterDataPipe buffer overflow, buffer size {self.buffer_size} is insufficient.')

    def get_next_element_by_instance(self, instance_id: int):
        if self._datapipe_iterator is None and self._child_stop[instance_id]:
            self._datapipe_iterator = iter(self.main_datapipe)
            self._snapshot_state = _SnapshotState.Iterating
            self.main_datapipe_exhausted = False
            for i in range(self.num_instances):
                self._child_stop[i] = False
        try:
            while not self._child_stop[instance_id]:
                if self.child_buffers[instance_id]:
                    self.current_buffer_usage -= 1
                    yield self.child_buffers[instance_id].popleft()
                else:
                    try:
                        yield self._find_next(instance_id)
                    except StopIteration:
                        self._child_stop[instance_id] = True
                        self.main_datapipe_exhausted = True
                        self._datapipe_iterator = None
        finally:
            self._child_stop[instance_id] = True
            if all(self._child_stop):
                self._datapipe_iterator = None
            if self.child_buffers[instance_id]:
                self._cleanup(instance_id)

    def is_every_instance_exhausted(self) -> bool:
        return self.main_datapipe_exhausted and all(self._child_stop)

    def get_length_by_instance(self, instance_id: int) -> int:
        raise TypeError

    def reset(self) -> None:
        self._datapipe_iterator = None
        self.current_buffer_usage = 0
        self.child_buffers = [deque() for _ in range(self.num_instances)]
        self._child_stop = [True for _ in range(self.num_instances)]
        self.main_datapipe_exhausted = False

    def __getstate__(self):
        state = (self.main_datapipe, self.num_instances, self.buffer_size, self.classifier_fn, self.drop_none, self._valid_iterator_id, self._number_of_samples_yielded)
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        self.main_datapipe, self.num_instances, self.buffer_size, self.classifier_fn, self.drop_none, self._valid_iterator_id, self._number_of_samples_yielded = state
        self._datapipe_iterator = None
        self.current_buffer_usage = 0
        self.child_buffers = [deque() for _ in range(self.num_instances)]
        self._child_stop = [True for _ in range(self.num_instances)]
        self.main_datapipe_exhausted = False

    def _cleanup(self, instance_id: Optional[int]=None):
        ids = range(self.num_instances) if instance_id is None else [instance_id]
        for i in ids:
            q = self.child_buffers[i]
            while q:
                d = q.popleft()
                StreamWrapper.close_streams(d)

    def __del__(self):
        self._cleanup()