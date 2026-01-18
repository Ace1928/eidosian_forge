from ...sage_helper import _within_sage
from .extended_matrix import ExtendedMatrix
def translate_PGL(self, m):
    """
        Let an extended PGL(2,C)-matrix or a PGL(2,C)-matrix act on the finite
        point.
        The matrix m should be an :class:`ExtendedMatrix` or a SageMath
        ``Matrix`` with coefficients in SageMath's ``ComplexIntervalField``::

            sage: from sage.all import *
            sage: pt = FinitePoint(CIF(1,2),RIF(3))
            sage: m = matrix([[CIF(0.25), CIF(1.2, 1)],[CIF(0.0), CIF(1.0)]])
            sage: pt.translate_PGL(m) # doctest: +NUMERIC12
            FinitePoint(1.4500000000000000? + 1.5000000000000000?*I, 0.75000000000000000?)

        """
    return self._translate(m, normalize_matrix=True)