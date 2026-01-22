import collections
import warnings
import weakref
from collections.abc import Mapping
from typing import Any, AnyStr, Optional, OrderedDict, Sequence, TypeVar
from scrapy.exceptions import ScrapyDeprecationWarning
class SequenceExclude:
    """Object to test if an item is NOT within some sequence."""

    def __init__(self, seq: Sequence):
        self.seq: Sequence = seq

    def __contains__(self, item: Any) -> bool:
        return item not in self.seq