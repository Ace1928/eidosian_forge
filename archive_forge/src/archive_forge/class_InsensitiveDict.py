from __future__ import annotations
import errno
import os
import sys
import warnings
from typing import AnyStr
from collections import OrderedDict
from typing import (
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
class InsensitiveDict(MutableMapping[str, _T]):
    """
    Dictionary, that has case-insensitive keys.

    Normally keys are retained in their original form when queried with
    .keys() or .items().  If initialized with preserveCase=0, keys are both
    looked up in lowercase and returned in lowercase by .keys() and .items().
    """
    '\n    Modified recipe at http://code.activestate.com/recipes/66315/ originally\n    contributed by Sami Hangaslammi.\n    '

    def __init__(self, dict=None, preserve=1):
        """
        Create an empty dictionary, or update from 'dict'.
        """
        super().__init__()
        self.data = {}
        self.preserve = preserve
        if dict:
            self.update(dict)

    def __delitem__(self, key):
        k = self._lowerOrReturn(key)
        del self.data[k]

    def _lowerOrReturn(self, key):
        if isinstance(key, bytes) or isinstance(key, str):
            return key.lower()
        else:
            return key

    def __getitem__(self, key):
        """
        Retrieve the value associated with 'key' (in any case).
        """
        k = self._lowerOrReturn(key)
        return self.data[k][1]

    def __setitem__(self, key, value):
        """
        Associate 'value' with 'key'. If 'key' already exists, but
        in different case, it will be replaced.
        """
        k = self._lowerOrReturn(key)
        self.data[k] = (key, value)

    def has_key(self, key):
        """
        Case insensitive test whether 'key' exists.
        """
        k = self._lowerOrReturn(key)
        return k in self.data
    __contains__ = has_key

    def _doPreserve(self, key):
        if not self.preserve and (isinstance(key, bytes) or isinstance(key, str)):
            return key.lower()
        else:
            return key

    def keys(self):
        """
        List of keys in their original case.
        """
        return list(self.iterkeys())

    def values(self):
        """
        List of values.
        """
        return list(self.itervalues())

    def items(self):
        """
        List of (key,value) pairs.
        """
        return list(self.iteritems())

    def get(self, key, default=None):
        """
        Retrieve value associated with 'key' or return default value
        if 'key' doesn't exist.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key, default):
        """
        If 'key' doesn't exist, associate it with the 'default' value.
        Return value associated with 'key'.
        """
        if not self.has_key(key):
            self[key] = default
        return self[key]

    def update(self, dict):
        """
        Copy (key,value) pairs from 'dict'.
        """
        for k, v in dict.items():
            self[k] = v

    def __repr__(self) -> str:
        """
        String representation of the dictionary.
        """
        items = ', '.join([f'{k!r}: {v!r}' for k, v in self.items()])
        return 'InsensitiveDict({%s})' % items

    def iterkeys(self):
        for v in self.data.values():
            yield self._doPreserve(v[0])
    __iter__ = iterkeys

    def itervalues(self):
        for v in self.data.values():
            yield v[1]

    def iteritems(self):
        for k, v in self.data.values():
            yield (self._doPreserve(k), v)
    _notFound = object()

    def pop(self, key, default=_notFound):
        """
        @see: L{dict.pop}
        @since: Twisted 21.2.0
        """
        try:
            return self.data.pop(self._lowerOrReturn(key))[1]
        except KeyError:
            if default is self._notFound:
                raise
            return default

    def popitem(self):
        i = self.items()[0]
        del self[i[0]]
        return i

    def clear(self):
        for k in self.keys():
            del self[k]

    def copy(self):
        return InsensitiveDict(self, self.preserve)

    def __len__(self):
        return len(self.data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            for k, v in self.items():
                if k not in other or other[k] != v:
                    return False
            return len(self) == len(other)
        else:
            return NotImplemented