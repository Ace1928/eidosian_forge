import re
import itertools
import textwrap
import functools
from jaraco.functools import compose, method_cache
from jaraco.context import ExceptionTrap
def trim_right(self, item):
    """
        Remove the item from the end of the set.

        >>> WordSet.parse('foo bar').trim_right('foo')
        ('foo', 'bar')
        >>> WordSet.parse('foo bar').trim_right('bar')
        ('foo',)
        >>> WordSet.parse('').trim_right('bar')
        ()
        """
    return self[:-1] if self and self[-1] == item else self