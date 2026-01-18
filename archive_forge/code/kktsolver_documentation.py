
    Returns a function handle "factor", which conforms to the CVXOPT
    custom KKT solver specifications:

        https://cvxopt.org/userguide/coneprog.html#exploiting-structure.

    For convenience, we provide a short outline for how this function works.

    First, we allocate workspace for use in "factor". The factor function is
    called with data (H, W). Once called, the factor function computes an LDL
    factorization of the 3 x 3 system:

        [ H           A'   G'*W^{-1}  ]
        [ A           0    0          ].
        [ W^{-T}*G    0   -I          ]

    Once that LDL factorization is computed, "factor" constructs another
    inner function, called "solve". The solve function uses the newly
    constructed LDL factorization to compute solutions to linear systems of
    the form

        [ H     A'   G'    ]   [ ux ]   [ bx ]
        [ A     0    0     ] * [ uy ] = [ by ].
        [ G     0   -W'*W  ]   [ uz ]   [ bz ]

    The factor function concludes by returning a reference to the solve function.

    Notes: In the 3 x 3 system, H is n x n, A is p x n, and G is N x n, where
    N = dims['l'] + sum(dims['q']) + sum( k**2 for k in dims['s'] ). For cone
    programs, H is the zero matrix.
    