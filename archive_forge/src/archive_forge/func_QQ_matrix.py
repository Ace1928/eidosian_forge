from sympy.core.numbers import igcd, ilcm
from sympy.core.symbol import Dummy
from sympy.polys.polyclasses import ANP
from sympy.polys.polytools import Poly
from sympy.polys.densetools import dup_clear_denoms
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.exceptions import DMBadInputError
from sympy.polys.matrices.normalforms import hermite_normal_form
from sympy.polys.polyerrors import CoercionFailed, UnificationFailed
from sympy.polys.polyutils import IntegerPowerable
from .exceptions import ClosureFailure, MissingUnityError, StructureError
from .utilities import AlgIntPowers, is_rat, get_num_denom
@property
def QQ_matrix(self):
    """
        :py:class:`~.DomainMatrix` over :ref:`QQ`, equal to
        ``self.matrix / self.denom``, and guaranteed to be dense.

        Explanation
        ===========

        Depending on how it is formed, a :py:class:`~.DomainMatrix` may have
        an internal representation that is sparse or dense. We guarantee a
        dense representation here, so that tests for equivalence of submodules
        always come out as expected.

        Examples
        ========

        >>> from sympy.polys import Poly, cyclotomic_poly, ZZ
        >>> from sympy.abc import x
        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.polys.numberfields.modules import PowerBasis
        >>> T = Poly(cyclotomic_poly(5, x))
        >>> A = PowerBasis(T)
        >>> B = A.submodule_from_matrix(3*DomainMatrix.eye(4, ZZ), denom=6)
        >>> C = A.submodule_from_matrix(DomainMatrix.eye(4, ZZ), denom=2)
        >>> print(B.QQ_matrix == C.QQ_matrix)
        True

        Returns
        =======

        :py:class:`~.DomainMatrix` over :ref:`QQ`

        """
    if self._QQ_matrix is None:
        self._QQ_matrix = (self.matrix / self.denom).to_dense()
    return self._QQ_matrix