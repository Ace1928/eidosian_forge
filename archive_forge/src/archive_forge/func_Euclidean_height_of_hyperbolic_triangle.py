from ...sage_helper import _within_sage
from .finite_point import *
from .extended_matrix import *
def Euclidean_height_of_hyperbolic_triangle(idealPoints):
    """
    Computes the Euclidean height of the hyperbolic triangle spanned by three
    ideal points. The height is the Euclidean radius of the hyperbolic plane
    containing the triangle or the Euclidean radius of one of its hyperbolic
    sides (if the projection onto the boundary is an obtuse triangle)::

        sage: from sage.all import CIF
        sage: z0 = CIF(0)
        sage: z1 = CIF(1)
        sage: Euclidean_height_of_hyperbolic_triangle([z0, z1, Infinity])
        [+infinity .. +infinity]

        sage: Euclidean_height_of_hyperbolic_triangle([z0, z1, CIF(0.5, 0.8)]) # doctest: +NUMERIC12
        0.556250000000000?

        sage: Euclidean_height_of_hyperbolic_triangle([z0, z1, CIF(10, 0.001)]) # doctest: +NUMERIC12
        5.000000025000000?

    """
    if Infinity in idealPoints:
        for idealPoint in idealPoints:
            if idealPoint != Infinity:
                RIF = idealPoint.real().parent()
                return RIF(sage.all.Infinity)
        raise Exception('What?')
    lengths = [abs(idealPoints[(i + 2) % 3] - idealPoints[(i + 1) % 3]) for i in range(3)]
    for i in range(3):
        if lengths[i] ** 2 > lengths[(i + 1) % 3] ** 2 + lengths[(i + 2) % 3] ** 2:
            return lengths[i] / 2
    length_total = sum(lengths)
    length_product = lengths[0] * lengths[1] * lengths[2]
    terms_product = (-lengths[0] + lengths[1] + lengths[2]) * (lengths[0] - lengths[1] + lengths[2]) * (lengths[0] + lengths[1] - lengths[2])
    return length_product / (terms_product * length_total).sqrt()