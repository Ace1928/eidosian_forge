from sympy.polys.polytools import Poly
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable
from sympy.utilities.decorator import public
from .basis import round_two, nilradical_mod_p
from .exceptions import StructureError
from .modules import ModuleEndomorphism, find_min_poly
from .utilities import coeff_search, supplement_a_subspace
def valuation(self, I):
    """
        Compute the $\\mathfrak{p}$-adic valuation of integral ideal I at this
        prime ideal.

        Parameters
        ==========

        I : :py:class:`~.Submodule`

        See Also
        ========

        prime_valuation

        """
    return prime_valuation(I, self)