import contextlib
from collections.abc import Iterable
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Type, Union
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter
from typing_extensions import Self, TypedDict, override
from lightning_fabric.utilities.data import sized_len
from lightning_fabric.utilities.types import _Stateful
from pytorch_lightning.utilities._pytree import _map_and_unflatten, _tree_flatten, tree_unflatten
class _Sequential(_ModeIterator):

    def __init__(self, iterables: List[Iterable], limits: Optional[List[Union[int, float]]]=None) -> None:
        super().__init__(iterables, limits)
        self._iterator_idx = 0

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        n = len(self.iterables)
        if n == 0 or self._iterator_idx >= n:
            raise StopIteration
        if self.limits is not None:
            while self.limits[self._iterator_idx] <= self._idx:
                self._use_next_iterator()
                if self._iterator_idx >= n:
                    raise StopIteration
        try:
            out = next(self.iterators[0])
        except StopIteration:
            self._use_next_iterator()
            return self.__next__()
        index = self._idx
        self._idx += 1
        return (out, index, self._iterator_idx)

    @override
    def __iter__(self) -> Self:
        self._iterator_idx = 0
        self._idx = 0
        self._load_current_iterator()
        return self

    @override
    def __len__(self) -> int:
        lengths = _get_iterables_lengths(self.iterables)
        if self.limits is not None:
            return sum((min(length, limit) for length, limit in zip(lengths, self.limits)))
        return sum(lengths)

    @override
    def reset(self) -> None:
        super().reset()
        self._iterator_idx = 0

    def _load_current_iterator(self) -> None:
        if self._iterator_idx < len(self.iterables):
            self.iterators = [iter(self.iterables[self._iterator_idx])]
        else:
            self.iterators = []

    def _use_next_iterator(self) -> None:
        self._iterator_idx += 1
        self._idx = 0
        self._load_current_iterator()