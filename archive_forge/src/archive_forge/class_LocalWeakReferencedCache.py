import collections
import warnings
import weakref
from collections.abc import Mapping
from typing import Any, AnyStr, Optional, OrderedDict, Sequence, TypeVar
from scrapy.exceptions import ScrapyDeprecationWarning
class LocalWeakReferencedCache(weakref.WeakKeyDictionary):
    """
    A weakref.WeakKeyDictionary implementation that uses LocalCache as its
    underlying data structure, making it ordered and capable of being size-limited.

    Useful for memoization, while avoiding keeping received
    arguments in memory only because of the cached references.

    Note: like LocalCache and unlike weakref.WeakKeyDictionary,
    it cannot be instantiated with an initial dictionary.
    """

    def __init__(self, limit: Optional[int]=None):
        super().__init__()
        self.data: LocalCache = LocalCache(limit=limit)

    def __setitem__(self, key: _KT, value: _VT) -> None:
        try:
            super().__setitem__(key, value)
        except TypeError:
            pass

    def __getitem__(self, key: _KT) -> Optional[_VT]:
        try:
            return super().__getitem__(key)
        except (TypeError, KeyError):
            return None