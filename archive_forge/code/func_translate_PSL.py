from ...sage_helper import _within_sage
from .extended_matrix import ExtendedMatrix
def translate_PSL(self, m):
    """
        Let an extended PSL(2,C)-matrix or a PSL(2,C)-matrix act on the finite
        point.
        The matrix m should be an :class:`ExtendedMatrix` or a SageMath ``Matrix``
        with coefficients in SageMath's ``ComplexIntervalField`` and have
        determinant 1::

            sage: from sage.all import *
            sage: pt = FinitePoint(CIF(1,2),RIF(3))
            sage: m = matrix([[CIF(0.5), CIF(2.4, 2)],[CIF(0.0), CIF(2.0)]])
            sage: pt.translate_PSL(m) # doctest: +NUMERIC12
            FinitePoint(1.4500000000000000? + 1.5000000000000000?*I, 0.75000000000000000?)
            sage: m = ExtendedMatrix(m, isOrientationReversing = True)
            sage: pt.translate_PSL(m) # doctest: +NUMERIC12
            FinitePoint(1.4500000000000000? + 0.50000000000000000?*I, 0.75000000000000000?)

        """
    return self._translate(m, normalize_matrix=False)