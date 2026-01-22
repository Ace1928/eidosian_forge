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
class CreateBoson(BosonicOperator, Creator):
    """
    Bosonic creation operator.
    """
    op_symbol = 'b+'

    def _dagger_(self):
        return AnnihilateBoson(self.state)

    def apply_operator(self, state):
        """
        Apply state to self if self is not symbolic and state is a FockStateKet, else
        multiply self by state.

        Examples
        ========

        >>> from sympy.physics.secondquant import B, Dagger, BKet
        >>> from sympy.abc import x, y, n
        >>> Dagger(B(x)).apply_operator(y)
        y*CreateBoson(x)
        >>> B(0).apply_operator(BKet((n,)))
        sqrt(n)*FockStateBosonKet((n - 1,))
        """
        if not self.is_symbolic and isinstance(state, FockStateKet):
            element = self.state
            amp = sqrt(state[element] + 1)
            return amp * state.up(element)
        else:
            return Mul(self, state)

    def __repr__(self):
        return 'CreateBoson(%s)' % self.state

    def _latex(self, printer):
        if self.state is S.Zero:
            return '{b^\\dagger_{0}}'
        else:
            return '{b^\\dagger_{%s}}' % self.state.name