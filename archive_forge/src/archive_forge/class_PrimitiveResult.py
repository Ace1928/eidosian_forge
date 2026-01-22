from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Generic, TypeVar
from .pub_result import PubResult
class PrimitiveResult(Generic[T]):
    """A container for multiple pub results and global metadata."""

    def __init__(self, pub_results: Iterable[T], metadata: dict[str, Any] | None=None):
        """
        Args:
            pub_results: Pub results.
            metadata: Metadata that is common to all pub results; metadata specific to particular
                pubs should be placed in their metadata fields. Keys are expected to be strings.
        """
        self._pub_results = list(pub_results)
        self._metadata = metadata or {}

    @property
    def metadata(self) -> dict[str, Any]:
        """The metadata of this primitive result."""
        return self._metadata

    def __getitem__(self, index) -> T:
        return self._pub_results[index]

    def __len__(self) -> int:
        return len(self._pub_results)

    def __repr__(self) -> str:
        return f'PrimitiveResult({self._pub_results}, metadata={self.metadata})'

    def __iter__(self) -> Iterable[T]:
        return iter(self._pub_results)