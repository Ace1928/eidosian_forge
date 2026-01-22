from .hyperbolicStructure import *
from .verificationError import *
from sage.all import RDF, pi, matrix, block_matrix, vector

    Finds unverified hyperbolic structure for an Mcomplex.

    >>> from snappy import Triangulation
    >>> from snappy.snap.t3mlite import Mcomplex
    >>> isosig = 'uLLvLALLQPAPAMcbehgilknmkonpoqrqrsttxxuvcaiauxawkkutxhqqw'
    >>> m = Mcomplex(Triangulation(isosig, remove_finite_vertices = False))
    >>> h = compute_approx_hyperbolic_structure_new(m)
    >>> all([ abs(s - _two_pi) < 1e-11 for s in h.angle_sums ])
    True

    