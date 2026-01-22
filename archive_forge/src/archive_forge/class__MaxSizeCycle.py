import contextlib
from collections.abc import Iterable
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Type, Union
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter
from typing_extensions import Self, TypedDict, override
from lightning_fabric.utilities.data import sized_len
from lightning_fabric.utilities.types import _Stateful
from pytorch_lightning.utilities._pytree import _map_and_unflatten, _tree_flatten, tree_unflatten
class _MaxSizeCycle(_ModeIterator):

    def __init__(self, iterables: List[Iterable], limits: Optional[List[Union[int, float]]]=None) -> None:
        super().__init__(iterables, limits)
        self._consumed: List[bool] = []

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        n = len(self.iterators)
        out = [None] * n
        for i in range(n):
            try:
                out[i] = next(self.iterators[i])
            except StopIteration:
                self._consumed[i] = True
                if all(self._consumed):
                    raise
                self.iterators[i] = iter(self.iterables[i])
                out[i] = next(self.iterators[i])
        index = self._idx
        self._idx += 1
        return (out, index, 0)

    @override
    def __iter__(self) -> Self:
        super().__iter__()
        self._consumed = [False] * len(self.iterables)
        return self

    @override
    def __len__(self) -> int:
        lengths = _get_iterables_lengths(self.iterables)
        if self.limits is not None:
            return max((min(length, limit) for length, limit in zip(lengths, self.limits)))
        return max(lengths)

    @override
    def reset(self) -> None:
        super().reset()
        self._consumed = []