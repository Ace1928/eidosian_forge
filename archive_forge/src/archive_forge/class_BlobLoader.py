from __future__ import annotations
import contextlib
import mimetypes
from abc import ABC, abstractmethod
from io import BufferedReader, BytesIO
from pathlib import PurePath
from typing import Any, Dict, Generator, Iterable, Mapping, Optional, Union, cast
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
class BlobLoader(ABC):
    """Abstract interface for blob loaders implementation.

    Implementer should be able to load raw content from a storage system according
    to some criteria and return the raw content lazily as a stream of blobs.
    """

    @abstractmethod
    def yield_blobs(self) -> Iterable[Blob]:
        """A lazy loader for raw data represented by LangChain's Blob object.

        Returns:
            A generator over blobs
        """