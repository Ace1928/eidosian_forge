from .computeApproxHyperbolicStructureNew import *
from .computeApproxHyperbolicStructureOrb import *
from .polishApproxHyperbolicStructure import *
from .krawczykCertifiedEdgeLengthsEngine import *
from .verifyHyperbolicStructureEngine import *
from .parseVertexGramMatrixFile import (
from snappy.snap.t3mlite import Mcomplex

    Computes a verified hyperbolic structure given a snappy.Triangulation.
    If all verification tests pass, the result is an instance of
    HyperbolicStructure with edge lengths being SageMath's
    RealIntervalField. Otherwise, raises an exception subclassed from
    VerificationError.

    The argument source specifies whether Orb ('orb') or a python-only
    implementation ('new') to find the initial unverified hyperbolic
    structure is used. It can also be a path to a vgm file containing
    the vertex gram matrices.

    The precision can be specified by the argument bits_prec.

        >>> from snappy import Triangulation
        >>> T = Triangulation('kLLLLPQkbcghihjijhjtsmnnnegufa', remove_finite_vertices = False)
        >>> bool(compute_verified_hyperbolic_structure(T, source = 'orb'))
        True
        >>> bool(compute_verified_hyperbolic_structure(T, source = 'new'))
        True

        >>> from sage.all import RealIntervalField, pi
        >>> RIF = RealIntervalField(212)
        >>> two_pi = RIF(2*pi)
        >>> h = compute_verified_hyperbolic_structure(T, source = 'orb', bits_prec = 212)
        >>> max([abs(d - two_pi) for d in h.angle_sums]) < RIF(1e-55)
        True

    