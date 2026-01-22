import warnings
from abc import ABC, abstractmethod
from collections import deque
import copy as copymodule
from typing import Any, Callable, Iterator, List, Literal, Optional, Sized, Tuple, TypeVar, Deque
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper, _check_unpickable_fn
@functional_datapipe('mux')
class MultiplexerIterDataPipe(IterDataPipe):
    """
    Yields one element at a time from each of the input Iterable DataPipes (functional name: ``mux``).

    As in, one element from the 1st input DataPipe, then one element from the 2nd DataPipe in the next iteration,
    and so on. It ends when the shortest input DataPipe is exhausted.

    Args:
        datapipes: Iterable DataPipes that will take turn to yield their elements, until the shortest DataPipe is exhausted

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1, dp2, dp3 = IterableWrapper(range(3)), IterableWrapper(range(10, 15)), IterableWrapper(range(20, 25))
        >>> list(dp1.mux(dp2, dp3))
        [0, 10, 20, 1, 11, 21, 2, 12, 22]
    """

    def __init__(self, *datapipes):
        self.datapipes = datapipes
        self.buffer: List = []

    def __iter__(self):
        iterators = [iter(x) for x in self.datapipes]
        while len(iterators):
            for it in iterators:
                try:
                    value = next(it)
                    self.buffer.append(value)
                except StopIteration:
                    self.buffer.clear()
                    return
            yield from self.buffer
            self.buffer.clear()

    def __len__(self):
        if all((isinstance(dp, Sized) for dp in self.datapipes)):
            return min((len(dp) for dp in self.datapipes)) * len(self.datapipes)
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")

    def reset(self) -> None:
        self.buffer = []

    def __getstate__(self):
        state = (self.datapipes, self._valid_iterator_id, self._number_of_samples_yielded)
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        self.datapipes, self._valid_iterator_id, self._number_of_samples_yielded = state
        self.buffer = []

    def __del__(self):
        self.buffer.clear()