from itertools import combinations
from sympy.combinatorics.graycode import GrayCode
Return indices of subset in superset in a list; the list is empty
        if all elements of ``subset`` are not in ``superset``.

        Examples
        ========

            >>> from sympy.combinatorics import Subset
            >>> superset = [1, 3, 2, 5, 4]
            >>> Subset.subset_indices([3, 2, 1], superset)
            [1, 2, 0]
            >>> Subset.subset_indices([1, 6], superset)
            []
            >>> Subset.subset_indices([], superset)
            []

        