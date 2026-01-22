from typing import Any, Iterator, List, Optional
from typing_extensions import override
from lightning_fabric.utilities.data import sized_len
from pytorch_lightning.utilities.combined_loader import _ITERATOR_RETURN, CombinedLoader
from pytorch_lightning.utilities.exceptions import MisconfigurationException
class _DataFetcher(Iterator):

    def __init__(self) -> None:
        self._combined_loader: Optional[CombinedLoader] = None
        self.iterator: Optional[Iterator] = None
        self.fetched: int = 0
        self.done: bool = False
        self.length: Optional[int] = None
        self._start_profiler = _profile_nothing
        self._stop_profiler = _profile_nothing

    @property
    def combined_loader(self) -> CombinedLoader:
        if self._combined_loader is None:
            raise MisconfigurationException(f'`{self.__class__.__name__}` should have been `setup` with a `CombinedLoader`.')
        return self._combined_loader

    def setup(self, combined_loader: CombinedLoader) -> None:
        self._combined_loader = combined_loader

    @override
    def __iter__(self) -> '_DataFetcher':
        self.iterator = iter(self.combined_loader)
        self.reset()
        return self

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        assert self.iterator is not None
        self._start_profiler()
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.done = True
            raise
        finally:
            self._stop_profiler()
        self.fetched += 1
        if self.length is not None:
            self.done = self.fetched >= self.length
        return batch

    def reset(self) -> None:
        self.fetched = 0
        if self._combined_loader is not None:
            self.length = sized_len(self.combined_loader)
            self.done = self.length == 0

    def teardown(self) -> None:
        self.reset()
        if self._combined_loader is not None:
            self._combined_loader.reset()
        self.iterator = None