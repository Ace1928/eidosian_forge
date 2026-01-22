from sympy.external.gmpy import MPQ
from sympy.polys.domains.groundtypes import SymPyRational
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
def algebraic_field(self, *extension, alias=None):
    """Returns an algebraic field, i.e. `\\mathbb{Q}(\\alpha, \\ldots)`.

        Parameters
        ==========

        *extension : One or more :py:class:`~.Expr`
            Generators of the extension. These should be expressions that are
            algebraic over `\\mathbb{Q}`.

        alias : str, :py:class:`~.Symbol`, None, optional (default=None)
            If provided, this will be used as the alias symbol for the
            primitive element of the returned :py:class:`~.AlgebraicField`.

        Returns
        =======

        :py:class:`~.AlgebraicField`
            A :py:class:`~.Domain` representing the algebraic field extension.

        Examples
        ========

        >>> from sympy import QQ, sqrt
        >>> QQ.algebraic_field(sqrt(2))
        QQ<sqrt(2)>
        """
    from sympy.polys.domains import AlgebraicField
    return AlgebraicField(self, *extension, alias=alias)