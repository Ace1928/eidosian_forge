from collections import defaultdict
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import has_dups
class BosonState(FockState):
    """
    Base class for FockStateBoson(Ket/Bra).
    """

    def up(self, i):
        """
        Performs the action of a creation operator.

        Examples
        ========

        >>> from sympy.physics.secondquant import BBra
        >>> b = BBra([1, 2])
        >>> b
        FockStateBosonBra((1, 2))
        >>> b.up(1)
        FockStateBosonBra((1, 3))
        """
        i = int(i)
        new_occs = list(self.args[0])
        new_occs[i] = new_occs[i] + S.One
        return self.__class__(new_occs)

    def down(self, i):
        """
        Performs the action of an annihilation operator.

        Examples
        ========

        >>> from sympy.physics.secondquant import BBra
        >>> b = BBra([1, 2])
        >>> b
        FockStateBosonBra((1, 2))
        >>> b.down(1)
        FockStateBosonBra((1, 1))
        """
        i = int(i)
        new_occs = list(self.args[0])
        if new_occs[i] == S.Zero:
            return S.Zero
        else:
            new_occs[i] = new_occs[i] - S.One
            return self.__class__(new_occs)