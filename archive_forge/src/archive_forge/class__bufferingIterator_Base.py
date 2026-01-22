import collections
import collections.abc
import logging
import sys
import textwrap
from abc import ABC
class _bufferingIterator_Base(collections.abc.Iterator, Generic[T], ABC):
    pass