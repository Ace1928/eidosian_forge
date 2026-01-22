from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j, wigner_9j
from sympy.printing.precedence import PRECEDENCE
class Wigner3j(Expr):
    """Class for the Wigner-3j symbols.

    Explanation
    ===========

    Wigner 3j-symbols are coefficients determined by the coupling of
    two angular momenta. When created, they are expressed as symbolic
    quantities that, for numerical parameters, can be evaluated using the
    ``.doit()`` method [1]_.

    Parameters
    ==========

    j1, m1, j2, m2, j3, m3 : Number, Symbol
        Terms determining the angular momentum of coupled angular momentum
        systems.

    Examples
    ========

    Declare a Wigner-3j coefficient and calculate its value

        >>> from sympy.physics.quantum.cg import Wigner3j
        >>> w3j = Wigner3j(6,0,4,0,2,0)
        >>> w3j
        Wigner3j(6, 0, 4, 0, 2, 0)
        >>> w3j.doit()
        sqrt(715)/143

    See Also
    ========

    CG: Clebsch-Gordan coefficients

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """
    is_commutative = True

    def __new__(cls, j1, m1, j2, m2, j3, m3):
        args = map(sympify, (j1, m1, j2, m2, j3, m3))
        return Expr.__new__(cls, *args)

    @property
    def j1(self):
        return self.args[0]

    @property
    def m1(self):
        return self.args[1]

    @property
    def j2(self):
        return self.args[2]

    @property
    def m2(self):
        return self.args[3]

    @property
    def j3(self):
        return self.args[4]

    @property
    def m3(self):
        return self.args[5]

    @property
    def is_symbolic(self):
        return not all((arg.is_number for arg in self.args))

    def _pretty(self, printer, *args):
        m = ((printer._print(self.j1), printer._print(self.m1)), (printer._print(self.j2), printer._print(self.m2)), (printer._print(self.j3), printer._print(self.m3)))
        hsep = 2
        vsep = 1
        maxw = [-1] * 3
        for j in range(3):
            maxw[j] = max([m[j][i].width() for i in range(2)])
        D = None
        for i in range(2):
            D_row = None
            for j in range(3):
                s = m[j][i]
                wdelta = maxw[j] - s.width()
                wleft = wdelta // 2
                wright = wdelta - wleft
                s = prettyForm(*s.right(' ' * wright))
                s = prettyForm(*s.left(' ' * wleft))
                if D_row is None:
                    D_row = s
                    continue
                D_row = prettyForm(*D_row.right(' ' * hsep))
                D_row = prettyForm(*D_row.right(s))
            if D is None:
                D = D_row
                continue
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))
            D = prettyForm(*D.below(D_row))
        D = prettyForm(*D.parens())
        return D

    def _latex(self, printer, *args):
        label = map(printer._print, (self.j1, self.j2, self.j3, self.m1, self.m2, self.m3))
        return '\\left(\\begin{array}{ccc} %s & %s & %s \\\\ %s & %s & %s \\end{array}\\right)' % tuple(label)

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError('Coefficients must be numerical')
        return wigner_3j(self.j1, self.j2, self.j3, self.m1, self.m2, self.m3)