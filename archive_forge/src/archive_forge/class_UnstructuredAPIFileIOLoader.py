import collections
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Any, Callable, Dict, Iterator, List, Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
class UnstructuredAPIFileIOLoader(UnstructuredFileIOLoader):
    """Load files using `Unstructured` API.

    By default, the loader makes a call to the hosted Unstructured API.
    If you are running the unstructured API locally, you can change the
    API rule by passing in the url parameter when you initialize the loader.
    The hosted Unstructured API requires an API key. See
    https://www.unstructured.io/api-key/ if you need to generate a key.

    You can run the loader in one of two modes: "single" and "elements".
    If you use "single" mode, the document will be returned as a single
    langchain Document object. If you use "elements" mode, the unstructured
    library will split the document into elements such as Title and NarrativeText.
    You can pass in additional unstructured kwargs after mode to apply
    different unstructured settings.

    Examples
    --------
    from langchain_community.document_loaders import UnstructuredAPIFileLoader

    with open("example.pdf", "rb") as f:
        loader = UnstructuredFileAPILoader(
            f, mode="elements", strategy="fast", api_key="MY_API_KEY",
        )
        docs = loader.load()

    References
    ----------
    https://unstructured-io.github.io/unstructured/bricks.html#partition
    https://www.unstructured.io/api-key/
    https://github.com/Unstructured-IO/unstructured-api
    """

    def __init__(self, file: Union[IO, Sequence[IO]], mode: str='single', url: str='https://api.unstructured.io/general/v0/general', api_key: str='', **unstructured_kwargs: Any):
        """Initialize with file path."""
        if isinstance(file, collections.abc.Sequence):
            validate_unstructured_version(min_unstructured_version='0.6.3')
        if file:
            validate_unstructured_version(min_unstructured_version='0.6.2')
        self.url = url
        self.api_key = api_key
        super().__init__(file=file, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        return get_elements_from_api(file=self.file, api_key=self.api_key, api_url=self.url, **self.unstructured_kwargs)