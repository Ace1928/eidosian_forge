from ..sage_helper import _within_sage, sage_method
from .. import SnapPy

        Given a representation rho to GL(R, n) and a rho-twisted
        1-cocycle, construct the representation to GL(R, n + 1)
        corresponding to the semidirect product.

        Note: Since we prefer to stick to left-actions only, unlike [HLK]
        this is the semidirect produce associated to the left action of
        GL(R, n) on V = R^n.  That is, pairs (v, A) with v in V and A in
        GL(R, n) where (v, A) * (w, B) = (v + A*w, A*B)::

           sage: G = Manifold('K12a169').fundamental_group()
           sage: A = matrix(GF(5), [[0, 4], [1, 4]])
           sage: rho = cyclic_rep(G, A)
           sage: cocycle = vector(GF(5), (0, 0, 1, 0))
           sage: rho_til = rho.semidirect_rep_from_twisted_cocycle(cocycle)
           sage: rho_til('abAB')
           [1 0 4]
           [0 1 1]
           [0 0 1]
        