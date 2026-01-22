from typing import Any, Iterator, List, Optional
from typing_extensions import override
from lightning_fabric.utilities.data import sized_len
from pytorch_lightning.utilities.combined_loader import _ITERATOR_RETURN, CombinedLoader
from pytorch_lightning.utilities.exceptions import MisconfigurationException
class _PrefetchDataFetcher(_DataFetcher):
    """This class is used to control batch fetching flow.

    Args:
        prefetch_batches: Number of batches to pre-fetch. Pre-fetching at least 1 batch is necessary to properly track
            whether a batch is the last one (available with :attr:`self.done`) when the length is not available. The
            value of this argument is ignored when the length is available.

    """

    def __init__(self, prefetch_batches: int=1) -> None:
        super().__init__()
        if prefetch_batches < 0:
            raise ValueError('`prefetch_batches` should at least be 0.')
        self.prefetch_batches = prefetch_batches
        self.batches: List[Any] = []

    @override
    def __iter__(self) -> '_PrefetchDataFetcher':
        super().__iter__()
        if self.length is not None:
            return self
        for _ in range(self.prefetch_batches):
            try:
                batch = super().__next__()
                self.batches.append(batch)
            except StopIteration:
                break
        return self

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        if self.batches:
            batch = self.batches.pop(0)
            try:
                self.batches.append(super().__next__())
            except StopIteration:
                self.done = not self.batches
        elif not self.done:
            batch = super().__next__()
        else:
            raise StopIteration
        return batch

    @override
    def reset(self) -> None:
        super().reset()
        self.batches = []