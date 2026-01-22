import sys
from collections.abc import MutableSet
from .compat import queue

        Implementation based on a doubly-linked link and an internal dictionary.
        This design gives :class:`OrderedSet` the same big-Oh running times as
        regular sets including O(1) adds, removes, and lookups as well as
        O(n) iteration.

        .. ADMONITION:: Implementation notes

                Runs on Python 2.6 or later (and runs on Python 3.0 or later
                without any modifications).

        :author: python@rcn.com (Raymond Hettinger)
        :url: http://code.activestate.com/recipes/576694/
        