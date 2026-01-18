from sympy.core import Basic
from sympy.core.containers import Tuple
from sympy.tensor.array import Array
from sympy.core.sympify import _sympify
from sympy.utilities.iterables import flatten, iterable
from sympy.utilities.misc import as_int
from collections import defaultdict
@classmethod
def unrank(self, rank, n):
    """Finds the unranked Prufer sequence.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> Prufer.unrank(0, 4)
        Prufer([0, 0])

        """
    n, rank = (as_int(n), as_int(rank))
    L = defaultdict(int)
    for i in range(n - 3, -1, -1):
        L[i] = rank % n
        rank = (rank - L[i]) // n
    return Prufer([L[i] for i in range(len(L))])