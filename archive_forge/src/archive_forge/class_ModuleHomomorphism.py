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
class ModuleHomomorphism:
    """A homomorphism from one module to another."""

    def __init__(self, domain, codomain, mapping):
        """
        Parameters
        ==========

        domain : :py:class:`~.Module`
            The domain of the mapping.

        codomain : :py:class:`~.Module`
            The codomain of the mapping.

        mapping : callable
            An arbitrary callable is accepted, but should be chosen so as
            to represent an actual module homomorphism. In particular, should
            accept elements of *domain* and return elements of *codomain*.

        Examples
        ========

        >>> from sympy import Poly, cyclotomic_poly
        >>> from sympy.polys.numberfields.modules import PowerBasis, ModuleHomomorphism
        >>> T = Poly(cyclotomic_poly(5))
        >>> A = PowerBasis(T)
        >>> B = A.submodule_from_gens([2*A(j) for j in range(4)])
        >>> phi = ModuleHomomorphism(A, B, lambda x: 6*x)
        >>> print(phi.matrix())  # doctest: +SKIP
        DomainMatrix([[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]], (4, 4), ZZ)

        """
        self.domain = domain
        self.codomain = codomain
        self.mapping = mapping

    def matrix(self, modulus=None):
        """
        Compute the matrix of this homomorphism.

        Parameters
        ==========

        modulus : int, optional
            A positive prime number $p$ if the matrix should be reduced mod
            $p$.

        Returns
        =======

        :py:class:`~.DomainMatrix`
            The matrix is over :ref:`ZZ`, or else over :ref:`GF(p)` if a
            modulus was given.

        """
        basis = self.domain.basis_elements()
        cols = [self.codomain.represent(self.mapping(elt)) for elt in basis]
        if not cols:
            return DomainMatrix.zeros((self.codomain.n, 0), ZZ).to_dense()
        M = cols[0].hstack(*cols[1:])
        if modulus:
            M = M.convert_to(FF(modulus))
        return M

    def kernel(self, modulus=None):
        """
        Compute a Submodule representing the kernel of this homomorphism.

        Parameters
        ==========

        modulus : int, optional
            A positive prime number $p$ if the kernel should be computed mod
            $p$.

        Returns
        =======

        :py:class:`~.Submodule`
            This submodule's generators span the kernel of this
            homomorphism over :ref:`ZZ`, or else over :ref:`GF(p)` if a
            modulus was given.

        """
        M = self.matrix(modulus=modulus)
        if modulus is None:
            M = M.convert_to(QQ)
        K = M.nullspace().convert_to(ZZ).transpose()
        return self.domain.submodule_from_matrix(K)