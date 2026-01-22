from typing import Any, Iterator, List, Optional
from typing_extensions import override
from lightning_fabric.utilities.data import sized_len
from pytorch_lightning.utilities.combined_loader import _ITERATOR_RETURN, CombinedLoader
from pytorch_lightning.utilities.exceptions import MisconfigurationException
class _DataFetcherWrapper(Iterator):

    def __init__(self, data_fetcher: _DataLoaderIterDataFetcher) -> None:
        self.data_fetcher = data_fetcher

    @property
    def done(self) -> bool:
        return self.data_fetcher.done

    @property
    def fetched(self) -> int:
        return self.data_fetcher.fetched

    @property
    def length(self) -> Optional[int]:
        return self.data_fetcher.length

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        fetcher = self.data_fetcher
        if fetcher.done:
            raise StopIteration
        batch, batch_idx, dataloader_idx = super(_DataLoaderIterDataFetcher, fetcher).__next__()
        fetcher._batch = batch
        fetcher._batch_idx = batch_idx
        fetcher._dataloader_idx = dataloader_idx
        return (batch, batch_idx, dataloader_idx)