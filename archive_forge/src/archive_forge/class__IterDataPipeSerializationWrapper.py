import functools
import pickle
from typing import Dict, Callable, Optional, TypeVar, Generic, Iterator
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.utils.common import (
from torch.utils.data.dataset import Dataset, IterableDataset
class _IterDataPipeSerializationWrapper(_DataPipeSerializationWrapper, IterDataPipe):

    def __init__(self, datapipe: IterDataPipe[T_co]):
        super().__init__(datapipe)
        self._datapipe_iter: Optional[Iterator[T_co]] = None

    def __iter__(self) -> '_IterDataPipeSerializationWrapper':
        self._datapipe_iter = iter(self._datapipe)
        return self

    def __next__(self) -> T_co:
        assert self._datapipe_iter is not None
        return next(self._datapipe_iter)