from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence, TypeVar, Union
from langchain_community.document_loaders.blob_loaders.schema import Blob, BlobLoader
def yield_blobs(self) -> Iterable[Blob]:
    """Yield blobs that match the requested pattern."""
    iterator = _make_iterator(length_func=self.count_matching_files, show_progress=self.show_progress)
    for path in iterator(self._yield_paths()):
        yield Blob.from_path(path)