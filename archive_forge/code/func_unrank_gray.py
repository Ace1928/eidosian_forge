from itertools import combinations
from sympy.combinatorics.graycode import GrayCode
@classmethod
def unrank_gray(self, rank, superset):
    """
        Gets the Gray code ordered subset of the specified rank.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> Subset.unrank_gray(4, ['a', 'b', 'c']).subset
        ['a', 'b']
        >>> Subset.unrank_gray(0, ['a', 'b', 'c']).subset
        []

        See Also
        ========

        iterate_graycode, rank_gray
        """
    graycode_bitlist = GrayCode.unrank(len(superset), rank)
    return Subset.subset_from_bitlist(superset, graycode_bitlist)