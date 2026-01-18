from itertools import combinations
from sympy.combinatorics.graycode import GrayCode
@classmethod
def unrank_binary(self, rank, superset):
    """
        Gets the binary ordered subset of the specified rank.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> Subset.unrank_binary(4, ['a', 'b', 'c', 'd']).subset
        ['b']

        See Also
        ========

        iterate_binary, rank_binary
        """
    bits = bin(rank)[2:].rjust(len(superset), '0')
    return Subset.subset_from_bitlist(superset, bits)