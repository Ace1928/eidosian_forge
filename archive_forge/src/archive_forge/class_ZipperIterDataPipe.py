import warnings
from abc import ABC, abstractmethod
from collections import deque
import copy as copymodule
from typing import Any, Callable, Iterator, List, Literal, Optional, Sized, Tuple, TypeVar, Deque
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper, _check_unpickable_fn
@functional_datapipe('zip')
class ZipperIterDataPipe(IterDataPipe[Tuple[T_co]]):
    """
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).

    The output is stopped as soon as the shortest input DataPipe is exhausted.

    Args:
        *datapipes: Iterable DataPipes being aggregated

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1, dp2, dp3 = IterableWrapper(range(5)), IterableWrapper(range(10, 15)), IterableWrapper(range(20, 25))
        >>> list(dp1.zip(dp2, dp3))
        [(0, 10, 20), (1, 11, 21), (2, 12, 22), (3, 13, 23), (4, 14, 24)]
    """
    datapipes: Tuple[IterDataPipe]

    def __init__(self, *datapipes: IterDataPipe):
        if not all((isinstance(dp, IterDataPipe) for dp in datapipes)):
            raise TypeError('All inputs are required to be `IterDataPipe` for `ZipIterDataPipe`.')
        super().__init__()
        self.datapipes = datapipes

    def __iter__(self) -> Iterator[Tuple[T_co]]:
        iterators = [iter(datapipe) for datapipe in self.datapipes]
        yield from zip(*iterators)

    def __len__(self) -> int:
        if all((isinstance(dp, Sized) for dp in self.datapipes)):
            return min((len(dp) for dp in self.datapipes))
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")