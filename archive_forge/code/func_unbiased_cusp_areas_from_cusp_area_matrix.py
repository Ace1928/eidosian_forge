from ..sage_helper import _within_sage
from ..math_basics import correct_min, is_RealIntervalFieldElement
def unbiased_cusp_areas_from_cusp_area_matrix(cusp_area_matrix):
    """

    Examples::

        sage: from sage.all import matrix, RIF
        sage: unbiased_cusp_areas_from_cusp_area_matrix(
        ...             matrix([[RIF(9.0,9.0005),RIF(6.0, 6.001)],
        ...                     [RIF(6.0,6.001 ),RIF(4.0, 4.001)]]))
        [3.00?, 2.000?]

        >>> from snappy.SnapPy import matrix
        >>> unbiased_cusp_areas_from_cusp_area_matrix(
        ...             matrix([[10.0, 40.0],
        ...                     [40.0, 20.0]]))
        [3.1622776601683795, 4.47213595499958]

    """
    if is_RealIntervalFieldElement(cusp_area_matrix[0, 0]):
        return _verified_unbiased_cusp_areas_from_cusp_area_matrix(cusp_area_matrix)
    return _unverified_unbiased_cusp_areas_from_cusp_area_matrix(cusp_area_matrix)