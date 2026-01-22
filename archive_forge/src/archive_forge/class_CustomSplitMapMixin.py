import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (
from mypy_extensions import trait
from black.comments import contains_pragma_comment
from black.lines import Line, append_leaves
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.rusty import Err, Ok, Result
from black.strings import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
@trait
class CustomSplitMapMixin:
    """
    This mixin class is used to map merged strings to a sequence of
    CustomSplits, which will then be used to re-split the strings iff none of
    the resultant substrings go over the configured max line length.
    """
    _Key: ClassVar = Tuple[StringID, str]
    _CUSTOM_SPLIT_MAP: ClassVar[Dict[_Key, Tuple[CustomSplit, ...]]] = defaultdict(tuple)

    @staticmethod
    def _get_key(string: str) -> 'CustomSplitMapMixin._Key':
        """
        Returns:
            A unique identifier that is used internally to map @string to a
            group of custom splits.
        """
        return (id(string), string)

    def add_custom_splits(self, string: str, custom_splits: Iterable[CustomSplit]) -> None:
        """Custom Split Map Setter Method

        Side Effects:
            Adds a mapping from @string to the custom splits @custom_splits.
        """
        key = self._get_key(string)
        self._CUSTOM_SPLIT_MAP[key] = tuple(custom_splits)

    def pop_custom_splits(self, string: str) -> List[CustomSplit]:
        """Custom Split Map Getter Method

        Returns:
            * A list of the custom splits that are mapped to @string, if any
              exist.
              OR
            * [], otherwise.

        Side Effects:
            Deletes the mapping between @string and its associated custom
            splits (which are returned to the caller).
        """
        key = self._get_key(string)
        custom_splits = self._CUSTOM_SPLIT_MAP[key]
        del self._CUSTOM_SPLIT_MAP[key]
        return list(custom_splits)

    def has_custom_splits(self, string: str) -> bool:
        """
        Returns:
            True iff @string is associated with a set of custom splits.
        """
        key = self._get_key(string)
        return key in self._CUSTOM_SPLIT_MAP