import warnings
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Iterator, List, Optional, Sized, TypeVar
import torch.utils.data.datapipes.iter.sharding
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DataChunk, IterDataPipe
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn
@functional_datapipe('unbatch')
class UnBatcherIterDataPipe(IterDataPipe):
    """
    Undos batching of data (functional name: ``unbatch``).

    In other words, it flattens the data up to the specified level within a batched DataPipe.

    Args:
        datapipe: Iterable DataPipe being un-batched
        unbatch_level: Defaults to ``1`` (only flattening the top level). If set to ``2``,
            it will flatten the top two levels, and ``-1`` will flatten the entire DataPipe.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper([[[0, 1], [2]], [[3, 4], [5]], [[6]]])
        >>> dp1 = source_dp.unbatch()
        >>> list(dp1)
        [[0, 1], [2], [3, 4], [5], [6]]
        >>> dp2 = source_dp.unbatch(unbatch_level=2)
        >>> list(dp2)
        [0, 1, 2, 3, 4, 5, 6]
    """

    def __init__(self, datapipe: IterDataPipe, unbatch_level: int=1):
        self.datapipe = datapipe
        self.unbatch_level = unbatch_level

    def __iter__(self):
        for element in self.datapipe:
            yield from self._dive(element, unbatch_level=self.unbatch_level)

    def _dive(self, element, unbatch_level):
        if unbatch_level < -1:
            raise ValueError('unbatch_level must be -1 or >= 0')
        if unbatch_level == -1:
            if isinstance(element, (list, DataChunk)):
                for item in element:
                    yield from self._dive(item, unbatch_level=-1)
            else:
                yield element
        elif unbatch_level == 0:
            yield element
        elif isinstance(element, (list, DataChunk)):
            for item in element:
                yield from self._dive(item, unbatch_level=unbatch_level - 1)
        else:
            raise IndexError(f'unbatch_level {self.unbatch_level} exceeds the depth of the DataPipe')