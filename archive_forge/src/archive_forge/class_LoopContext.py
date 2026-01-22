import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
class LoopContext:
    """A magic loop variable.
    Automatically accessible in any ``% for`` block.

    See the section :ref:`loop_context` for usage
    notes.

    :attr:`parent` -> :class:`.LoopContext` or ``None``
        The parent loop, if one exists.
    :attr:`index` -> `int`
        The 0-based iteration count.
    :attr:`reverse_index` -> `int`
        The number of iterations remaining.
    :attr:`first` -> `bool`
        ``True`` on the first iteration, ``False`` otherwise.
    :attr:`last` -> `bool`
        ``True`` on the last iteration, ``False`` otherwise.
    :attr:`even` -> `bool`
        ``True`` when ``index`` is even.
    :attr:`odd` -> `bool`
        ``True`` when ``index`` is odd.
    """

    def __init__(self, iterable):
        self._iterable = iterable
        self.index = 0
        self.parent = None

    def __iter__(self):
        for i in self._iterable:
            yield i
            self.index += 1

    @util.memoized_instancemethod
    def __len__(self):
        return len(self._iterable)

    @property
    def reverse_index(self):
        return len(self) - self.index - 1

    @property
    def first(self):
        return self.index == 0

    @property
    def last(self):
        return self.index == len(self) - 1

    @property
    def even(self):
        return not self.odd

    @property
    def odd(self):
        return bool(self.index % 2)

    def cycle(self, *values):
        """Cycle through values as the loop progresses."""
        if not values:
            raise ValueError('You must provide values to cycle through')
        return values[self.index % len(values)]