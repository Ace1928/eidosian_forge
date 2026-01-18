from sympy.core import Basic
from sympy.core.containers import Tuple
from sympy.tensor.array import Array
from sympy.core.sympify import _sympify
from sympy.utilities.iterables import flatten, iterable
from sympy.utilities.misc import as_int
from collections import defaultdict
Generates the Prufer sequence that is -delta before the current one.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> a = Prufer([[0, 1], [1, 2], [2, 3], [1, 4]])
        >>> a.rank
        36
        >>> b = a.prev()
        >>> b
        Prufer([1, 2, 0])
        >>> b.rank
        35

        See Also
        ========

        prufer_rank, rank, next, size

        