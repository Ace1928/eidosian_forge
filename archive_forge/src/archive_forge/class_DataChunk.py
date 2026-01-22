import functools
import pickle
from typing import Dict, Callable, Optional, TypeVar, Generic, Iterator
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.utils.common import (
from torch.utils.data.dataset import Dataset, IterableDataset
class DataChunk(list, Generic[T]):

    def __init__(self, items):
        super().__init__(items)
        self.items = items

    def as_str(self, indent=''):
        res = indent + '[' + ', '.join((str(i) for i in iter(self))) + ']'
        return res

    def __iter__(self) -> Iterator[T]:
        yield from super().__iter__()

    def raw_iterator(self) -> T:
        yield from self.items