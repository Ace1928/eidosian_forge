from __future__ import absolute_import
import itertools
import sys
from weakref import ref
class _Info(object):
    __slots__ = ('weakref', 'func', 'args', 'kwargs', 'atexit', 'index')