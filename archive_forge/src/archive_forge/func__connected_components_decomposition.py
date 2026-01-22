from sympy.utilities.iterables import \
from .common import NonSquareMatrixError
def _connected_components_decomposition(M):
    """Decomposes a square matrix into block diagonal form only
    using the permutations.

    Explanation
    ===========

    The decomposition is in a form of $A = P^{-1} B P$ where $P$ is a
    permutation matrix and $B$ is a block diagonal matrix.

    Returns
    =======

    P, B : PermutationMatrix, BlockDiagMatrix
        *P* is a permutation matrix for the similarity transform
        as in the explanation. And *B* is the block diagonal matrix of
        the result of the permutation.

        If you would like to get the diagonal blocks from the
        BlockDiagMatrix, see
        :meth:`~sympy.matrices.expressions.blockmatrix.BlockDiagMatrix.get_diag_blocks`.

    Examples
    ========

    >>> from sympy import Matrix, pprint
    >>> A = Matrix([
    ...     [66, 0, 0, 68, 0, 0, 0, 0, 67],
    ...     [0, 55, 0, 0, 0, 0, 54, 53, 0],
    ...     [0, 0, 0, 0, 1, 2, 0, 0, 0],
    ...     [86, 0, 0, 88, 0, 0, 0, 0, 87],
    ...     [0, 0, 10, 0, 11, 12, 0, 0, 0],
    ...     [0, 0, 20, 0, 21, 22, 0, 0, 0],
    ...     [0, 45, 0, 0, 0, 0, 44, 43, 0],
    ...     [0, 35, 0, 0, 0, 0, 34, 33, 0],
    ...     [76, 0, 0, 78, 0, 0, 0, 0, 77]])

    >>> P, B = A.connected_components_decomposition()
    >>> pprint(P)
    PermutationMatrix((1 3)(2 8 5 7 4 6))
    >>> pprint(B)
    [[66  68  67]                            ]
    [[          ]                            ]
    [[86  88  87]       0             0      ]
    [[          ]                            ]
    [[76  78  77]                            ]
    [                                        ]
    [              [55  54  53]              ]
    [              [          ]              ]
    [     0        [45  44  43]       0      ]
    [              [          ]              ]
    [              [35  34  33]              ]
    [                                        ]
    [                            [0   1   2 ]]
    [                            [          ]]
    [     0             0        [10  11  12]]
    [                            [          ]]
    [                            [20  21  22]]

    >>> P = P.as_explicit()
    >>> B = B.as_explicit()
    >>> P.T*B*P == A
    True

    Notes
    =====

    This problem corresponds to the finding of the connected components
    of a graph, when a matrix is viewed as a weighted graph.
    """
    from sympy.combinatorics.permutations import Permutation
    from sympy.matrices.expressions.blockmatrix import BlockDiagMatrix
    from sympy.matrices.expressions.permutation import PermutationMatrix
    iblocks = M.connected_components()
    p = Permutation(flatten(iblocks))
    P = PermutationMatrix(p)
    blocks = []
    for b in iblocks:
        blocks.append(M[b, b])
    B = BlockDiagMatrix(*blocks)
    return (P, B)