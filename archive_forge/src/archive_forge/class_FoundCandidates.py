import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Set, Tuple
from pip._vendor.packaging.version import _BaseVersion
from .base import Candidate
class FoundCandidates(SequenceCandidate):
    """A lazy sequence to provide candidates to the resolver.

    The intended usage is to return this from `find_matches()` so the resolver
    can iterate through the sequence multiple times, but only access the index
    page when remote packages are actually needed. This improve performances
    when suitable candidates are already installed on disk.
    """

    def __init__(self, get_infos: Callable[[], Iterator[IndexCandidateInfo]], installed: Optional[Candidate], prefers_installed: bool, incompatible_ids: Set[int]):
        self._get_infos = get_infos
        self._installed = installed
        self._prefers_installed = prefers_installed
        self._incompatible_ids = incompatible_ids

    def __getitem__(self, index: Any) -> Any:
        raise NotImplementedError("don't do this")

    def __iter__(self) -> Iterator[Candidate]:
        infos = self._get_infos()
        if not self._installed:
            iterator = _iter_built(infos)
        elif self._prefers_installed:
            iterator = _iter_built_with_prepended(self._installed, infos)
        else:
            iterator = _iter_built_with_inserted(self._installed, infos)
        return (c for c in iterator if id(c) not in self._incompatible_ids)

    def __len__(self) -> int:
        raise NotImplementedError("don't do this")

    @functools.lru_cache(maxsize=1)
    def __bool__(self) -> bool:
        if self._prefers_installed and self._installed:
            return True
        return any(self)