import contextlib as cl
import functools
import itertools as it
import operator
import os.path as op
import sqlite3
import tempfile
import time
from .core import DEFAULT_SETTINGS, ENOVAL, Cache, Disk, Timeout
from .persistent import Deque, Index
Return Index with given `name` in subdirectory.

        >>> cache = FanoutCache()
        >>> index = cache.index('test')
        >>> index['abc'] = 123
        >>> index['def'] = 456
        >>> index['ghi'] = 789
        >>> index.popitem()
        ('ghi', 789)
        >>> del index['abc']
        >>> len(index)
        1
        >>> index['def']
        456

        :param str name: subdirectory name for Index
        :return: Index with given name

        