import collections.abc
import contextlib
import sys
import textwrap
import weakref
from abc import ABC
from types import TracebackType
from weakref import ReferenceType
from debian._deb822_repro._util import (combine_into_replacement, BufferingIterator,
from debian._deb822_repro.formatter import (
from debian._deb822_repro.tokens import (
from debian._deb822_repro.types import AmbiguousDeb822FieldKeyError, SyntaxOrParseError
from debian._util import (
def _unpack_key(item, raise_if_indexed=False):
    index = None
    name_token = None
    if isinstance(item, tuple):
        key, index = item
        if raise_if_indexed:
            if index != 0:
                msg = 'Cannot resolve key "{key}" with index {index}. The key is not indexed'
                raise KeyError(msg.format(key=key, index=index))
            index = None
        key = _strI(key)
    else:
        index = None
        if isinstance(item, Deb822FieldNameToken):
            name_token = item
            key = name_token.text
        else:
            key = _strI(item)
    return (key, index, name_token)