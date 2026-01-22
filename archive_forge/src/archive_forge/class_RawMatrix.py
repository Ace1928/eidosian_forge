from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import connected_components
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer, Rational
from sympy.matrices.dense import MutableDenseMatrix
from sympy.polys.domains import ZZ, QQ
from sympy.polys.domains import EX
from sympy.polys.rings import sring
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.domainmatrix import DomainMatrix
class RawMatrix(MutableDenseMatrix):
    """
    .. deprecated:: 1.9

       This class fundamentally is broken by design. Use ``DomainMatrix`` if
       you want a matrix over the polys domains or ``Matrix`` for a matrix
       with ``Expr`` elements. The ``RawMatrix`` class will be removed/broken
       in future in order to reestablish the invariant that the elements of a
       Matrix should be of type ``Expr``.

    """
    _sympify = staticmethod(lambda x: x)

    def __init__(self, *args, **kwargs):
        sympy_deprecation_warning('\n            The RawMatrix class is deprecated. Use either DomainMatrix or\n            Matrix instead.\n            ', deprecated_since_version='1.9', active_deprecations_target='deprecated-rawmatrix')
        domain = ZZ
        for i in range(self.rows):
            for j in range(self.cols):
                val = self[i, j]
                if getattr(val, 'is_Poly', False):
                    K = val.domain[val.gens]
                    val_sympy = val.as_expr()
                elif hasattr(val, 'parent'):
                    K = val.parent()
                    val_sympy = K.to_sympy(val)
                elif isinstance(val, (int, Integer)):
                    K = ZZ
                    val_sympy = sympify(val)
                elif isinstance(val, Rational):
                    K = QQ
                    val_sympy = val
                else:
                    for K in (ZZ, QQ):
                        if K.of_type(val):
                            val_sympy = K.to_sympy(val)
                            break
                    else:
                        raise TypeError
                domain = domain.unify(K)
                self[i, j] = val_sympy
        self.ring = domain