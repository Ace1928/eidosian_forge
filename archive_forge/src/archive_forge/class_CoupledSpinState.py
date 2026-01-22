from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.simplify.simplify import simplify
from sympy.matrices import zeros
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import pretty_symbol
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.operator import (HermitianOperator, Operator,
from sympy.physics.quantum.state import Bra, Ket, State
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.hilbert import ComplexSpace, DirectSumHilbertSpace
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.qapply import qapply
class CoupledSpinState(SpinState):
    """Base class for coupled angular momentum states."""

    def __new__(cls, j, m, jn, *jcoupling):
        SpinState(j, m)
        if len(jcoupling) == 0:
            jcoupling = []
            for n in range(2, len(jn)):
                jcoupling.append((1, n, Add(*[jn[i] for i in range(n)])))
            jcoupling.append((1, len(jn), j))
        elif len(jcoupling) == 1:
            jcoupling = jcoupling[0]
        else:
            raise TypeError('CoupledSpinState only takes 3 or 4 arguments, got: %s' % (len(jcoupling) + 3))
        if not isinstance(jn, (list, tuple, Tuple)):
            raise TypeError('jn must be Tuple, list or tuple, got %s' % jn.__class__.__name__)
        if not isinstance(jcoupling, (list, tuple, Tuple)):
            raise TypeError('jcoupling must be Tuple, list or tuple, got %s' % jcoupling.__class__.__name__)
        if not all((isinstance(term, (list, tuple, Tuple)) for term in jcoupling)):
            raise TypeError('All elements of jcoupling must be list, tuple or Tuple')
        if not len(jn) - 1 == len(jcoupling):
            raise ValueError('jcoupling must have length of %d, got %d' % (len(jn) - 1, len(jcoupling)))
        if not all((len(x) == 3 for x in jcoupling)):
            raise ValueError('All elements of jcoupling must have length 3')
        j = sympify(j)
        m = sympify(m)
        jn = Tuple(*[sympify(ji) for ji in jn])
        jcoupling = Tuple(*[Tuple(sympify(n1), sympify(n2), sympify(ji)) for n1, n2, ji in jcoupling])
        if any((2 * ji != int(2 * ji) for ji in jn if ji.is_number)):
            raise ValueError('All elements of jn must be integer or half-integer, got: %s' % jn)
        if any((n1 != int(n1) or n2 != int(n2) for n1, n2, _ in jcoupling)):
            raise ValueError('Indices in jcoupling must be integers')
        if any((n1 < 1 or n2 < 1 or n1 > len(jn) or (n2 > len(jn)) for n1, n2, _ in jcoupling)):
            raise ValueError('Indices must be between 1 and the number of coupled spin spaces')
        if any((2 * ji != int(2 * ji) for _, _, ji in jcoupling if ji.is_number)):
            raise ValueError('All coupled j values in coupling scheme must be integer or half-integer')
        coupled_n, coupled_jn = _build_coupled(jcoupling, len(jn))
        jvals = list(jn)
        for n, (n1, n2) in enumerate(coupled_n):
            j1 = jvals[min(n1) - 1]
            j2 = jvals[min(n2) - 1]
            j3 = coupled_jn[n]
            if sympify(j1).is_number and sympify(j2).is_number and sympify(j3).is_number:
                if j1 + j2 < j3:
                    raise ValueError('All couplings must have j1+j2 >= j3, in coupling number %d got j1,j2,j3: %d,%d,%d' % (n + 1, j1, j2, j3))
                if abs(j1 - j2) > j3:
                    raise ValueError('All couplings must have |j1+j2| <= j3, in coupling number %d got j1,j2,j3: %d,%d,%d' % (n + 1, j1, j2, j3))
                if int(j1 + j2) == j1 + j2:
                    pass
            jvals[min(n1 + n2) - 1] = j3
        if len(jcoupling) > 0 and jcoupling[-1][2] != j:
            raise ValueError('Last j value coupled together must be the final j of the state')
        return State.__new__(cls, j, m, jn, jcoupling)

    def _print_label(self, printer, *args):
        label = [printer._print(self.j), printer._print(self.m)]
        for i, ji in enumerate(self.jn, start=1):
            label.append('j%d=%s' % (i, printer._print(ji)))
        for jn, (n1, n2) in zip(self.coupled_jn[:-1], self.coupled_n[:-1]):
            label.append('j(%s)=%s' % (','.join((str(i) for i in sorted(n1 + n2))), printer._print(jn)))
        return ','.join(label)

    def _print_label_pretty(self, printer, *args):
        label = [self.j, self.m]
        for i, ji in enumerate(self.jn, start=1):
            symb = 'j%d' % i
            symb = pretty_symbol(symb)
            symb = prettyForm(symb + '=')
            item = prettyForm(*symb.right(printer._print(ji)))
            label.append(item)
        for jn, (n1, n2) in zip(self.coupled_jn[:-1], self.coupled_n[:-1]):
            n = ','.join((pretty_symbol('j%d' % i)[-1] for i in sorted(n1 + n2)))
            symb = prettyForm('j' + n + '=')
            item = prettyForm(*symb.right(printer._print(jn)))
            label.append(item)
        return self._print_sequence_pretty(label, self._label_separator, printer, *args)

    def _print_label_latex(self, printer, *args):
        label = [printer._print(self.j, *args), printer._print(self.m, *args)]
        for i, ji in enumerate(self.jn, start=1):
            label.append('j_{%d}=%s' % (i, printer._print(ji, *args)))
        for jn, (n1, n2) in zip(self.coupled_jn[:-1], self.coupled_n[:-1]):
            n = ','.join((str(i) for i in sorted(n1 + n2)))
            label.append('j_{%s}=%s' % (n, printer._print(jn, *args)))
        return self._label_separator.join(label)

    @property
    def jn(self):
        return self.label[2]

    @property
    def coupling(self):
        return self.label[3]

    @property
    def coupled_jn(self):
        return _build_coupled(self.label[3], len(self.label[2]))[1]

    @property
    def coupled_n(self):
        return _build_coupled(self.label[3], len(self.label[2]))[0]

    @classmethod
    def _eval_hilbert_space(cls, label):
        j = Add(*label[2])
        if j.is_number:
            return DirectSumHilbertSpace(*[ComplexSpace(x) for x in range(int(2 * j + 1), 0, -2)])
        else:
            return ComplexSpace(2 * j + 1)

    def _represent_coupled_base(self, **options):
        evect = self.uncoupled_class()
        if not self.j.is_number:
            raise ValueError('State must not have symbolic j value to represent')
        if not self.hilbert_space.dimension.is_number:
            raise ValueError('State must not have symbolic j values to represent')
        result = zeros(self.hilbert_space.dimension, 1)
        if self.j == int(self.j):
            start = self.j ** 2
        else:
            start = (2 * self.j - 1) * (1 + 2 * self.j) / 4
        result[start:start + 2 * self.j + 1, 0] = evect(self.j, self.m)._represent_base(**options)
        return result

    def _eval_rewrite_as_Jx(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jx, JxBraCoupled, **options)
        return self._rewrite_basis(Jx, JxKetCoupled, **options)

    def _eval_rewrite_as_Jy(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jy, JyBraCoupled, **options)
        return self._rewrite_basis(Jy, JyKetCoupled, **options)

    def _eval_rewrite_as_Jz(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jz, JzBraCoupled, **options)
        return self._rewrite_basis(Jz, JzKetCoupled, **options)