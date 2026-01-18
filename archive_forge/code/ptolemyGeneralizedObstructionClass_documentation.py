from . import matrix
from .polynomial import Polynomial
from ..pari import pari

    Represents an obstruction cocycle of a PSL(n,C) representation in
    H^2(M,partial M;Z/n).

    >>> from snappy import Manifold
    >>> M = Manifold("m004")

    Create an obstruction class, this has to be in the kernel of d^2:

    >>> c = PtolemyGeneralizedObstructionClass([2,0,0,1])

    For better accounting, give it an index:

    >>> c = PtolemyGeneralizedObstructionClass([2,0,0,1], index = 1)

    Get corresponding ptolemy variety:

    >>> p = M.ptolemy_variety(N=3, obstruction_class=c)

    Canonical filename base:

    >>> p.filename_base()
    'm004__sl3_c1'

    Now pick something not in the kernel:

    >>> c = PtolemyGeneralizedObstructionClass([1,0,0,1])
    >>> p = M.ptolemy_variety(N=3, obstruction_class=c)
    Traceback (most recent call last):
    ...
    AssertionError: PtolemyGeneralizedObstructionClass not in kernel of d2

    