import sys
import warnings
from itertools import chain
from .sortedlist import SortedList, recursive_repr
from .sortedset import SortedSet
Lookup value at `index` in sorted values view.

        ``siv.__getitem__(index)`` <==> ``siv[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sd = SortedDict({'a': 1, 'b': 2, 'c': 3})
        >>> svv = sd.values()
        >>> svv[0]
        1
        >>> svv[-1]
        3
        >>> svv[:]
        [1, 2, 3]
        >>> svv[100]
        Traceback (most recent call last):
          ...
        IndexError: list index out of range

        :param index: integer or slice for indexing
        :return: value or list of values
        :raises IndexError: if index out of range

        