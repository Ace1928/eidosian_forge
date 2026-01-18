from sympy.combinatorics.permutations import Permutation, Perm
from sympy.combinatorics.tensor_can import (perm_af_direct_product, dummy_sgs,
from sympy.combinatorics.testutil import canonicalize_naive, graph_certificate
from sympy.testing.pytest import skip, XFAIL

    The following tests in test_riemann_invariants and in
    test_riemann_invariants1 have been checked using xperm.c from XPerm in
    in [1] and with an older version contained in [2]

    [1] xperm.c part of xPerm written by J. M. Martin-Garcia
        http://www.xact.es/index.html
    [2] test_xperm.cc in cadabra by Kasper Peeters, http://cadabra.phi-sci.com/
    