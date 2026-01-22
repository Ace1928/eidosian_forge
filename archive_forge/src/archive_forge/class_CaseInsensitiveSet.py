from __future__ import print_function, unicode_literals
import itertools
from collections import OrderedDict, deque
from functools import wraps
from types import GeneratorType
from six.moves import zip_longest
from .py3compat import fix_unicode_literals_in_doctest
@fix_unicode_literals_in_doctest
class CaseInsensitiveSet(MutableSet):
    """A very basic case-insensitive set.

    >>> s = CaseInsensitiveSet()
    >>> len(s)
    0
    >>> 'a' in s
    False
    >>> list(CaseInsensitiveSet(['aaa', 'Aaa', 'AAA']))
    [u'aaa']
    >>> s = CaseInsensitiveSet(['Aaa', 'Bbb'])
    >>> s
    CaseInsensitiveSet([u'Aaa', u'Bbb'])
    >>> s.lower()
    CaseInsensitiveSet([u'aaa', u'bbb'])
    >>> len(s)
    2
    >>> 'aaa' in s
    True
    >>> 'Aaa' in s
    True
    >>> 'AAA' in s
    True
    >>> 'bbb' in s
    True
    >>> 'Bbb' in s
    True
    >>> 'abc' in s
    False
    >>> s.add('ccc')
    >>> len(s)
    3
    >>> 'aaa' in s
    True
    >>> 'ccc' in s
    True
    >>> s.remove('AAA')
    >>> len(s)
    2
    >>> 'aaa' in s
    False

    >>> bool(CaseInsensitiveSet(['a']))
    True
    >>> bool(CaseInsensitiveSet([]))
    False
    >>> bool(CaseInsensitiveSet())
    False

    """

    def __init__(self, iterable=()):
        self._set = set()
        self._keys = dict()
        for item in iterable:
            self.add(item)

    def __contains__(self, key):
        return key.lower() in self._set

    def __iter__(self):
        return iter(self._set)

    def __len__(self):
        return len(self._set)

    def __repr__(self):
        """A caselessDict version of __repr__ """
        return '{0}({1})'.format(type(self).__name__, repr(sorted(self._keys.values())))

    def add(self, key):
        key_lower = key.lower()
        self._set.add(key_lower)
        self._keys[key_lower] = key

    def discard(self, key):
        key_lower = key.lower()
        self._set.discard(key_lower)
        self._keys.pop(key_lower, None)

    def get_canonical_key(self, key):
        return self._keys[key.lower()]

    def lower(self):
        return type(self)(self._set)