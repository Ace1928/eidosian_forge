import sys
import re
import warnings
import io
import collections
import collections.abc
import contextlib
import weakref
from . import ElementPath
fromstring = XML
class IterParseIterator(collections.abc.Iterator):
    __next__ = iterator(source).__next__

    def __del__(self):
        if close_source:
            source.close()