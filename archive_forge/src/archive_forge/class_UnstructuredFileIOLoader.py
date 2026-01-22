import collections
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Any, Callable, Dict, Iterator, List, Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
class UnstructuredFileIOLoader(UnstructuredBaseLoader):
    """Load files using `Unstructured`.

    The file loader
    uses the unstructured partition function and will automatically detect the file
    type. You can run the loader in one of two modes: "single" and "elements".
    If you use "single" mode, the document will be returned as a single
    langchain Document object. If you use "elements" mode, the unstructured
    library will split the document into elements such as Title and NarrativeText.
    You can pass in additional unstructured kwargs after mode to apply
    different unstructured settings.

    Examples
    --------
    from langchain_community.document_loaders import UnstructuredFileIOLoader

    with open("example.pdf", "rb") as f:
        loader = UnstructuredFileIOLoader(
            f, mode="elements", strategy="fast",
        )
        docs = loader.load()


    References
    ----------
    https://unstructured-io.github.io/unstructured/bricks.html#partition
    """

    def __init__(self, file: Union[IO, Sequence[IO]], mode: str='single', **unstructured_kwargs: Any):
        """Initialize with file path."""
        self.file = file
        super().__init__(mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.auto import partition
        return partition(file=self.file, **self.unstructured_kwargs)

    def _get_metadata(self) -> dict:
        return {}