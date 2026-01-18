from __future__ import annotations
from typing import Any
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.cg import CG, Wigner3j, Wigner6j, Wigner9j
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.gate import CGate, CNotGate, IdentityGate, UGate, XGate
from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace, HilbertSpace, L2
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.operator import Operator, OuterProduct, DifferentialOperator
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.qubit import Qubit, IntQubit
from sympy.physics.quantum.spin import Jz, J2, JzBra, JzBraCoupled, JzKet, JzKetCoupled, Rotation, WignerD
from sympy.physics.quantum.state import Bra, Ket, TimeDepBra, TimeDepKet
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.sho1d import RaisingOp
from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import oo
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.matrices.dense import Matrix
from sympy.sets.sets import Interval
from sympy.testing.pytest import XFAIL
from sympy.physics.quantum.spin import JzOp
from sympy.printing import srepr
from sympy.printing.pretty import pretty as xpretty
from sympy.printing.latex import latex
def test_operator():
    a = Operator('A')
    b = Operator('B', Symbol('t'), S.Half)
    inv = a.inv()
    f = Function('f')
    x = symbols('x')
    d = DifferentialOperator(Derivative(f(x), x), f(x))
    op = OuterProduct(Ket(), Bra())
    assert str(a) == 'A'
    assert pretty(a) == 'A'
    assert upretty(a) == 'A'
    assert latex(a) == 'A'
    sT(a, "Operator(Symbol('A'))")
    assert str(inv) == 'A**(-1)'
    ascii_str = ' -1\nA  '
    ucode_str = ' -1\nA  '
    assert pretty(inv) == ascii_str
    assert upretty(inv) == ucode_str
    assert latex(inv) == 'A^{-1}'
    sT(inv, "Pow(Operator(Symbol('A')), Integer(-1))")
    assert str(d) == 'DifferentialOperator(Derivative(f(x), x),f(x))'
    ascii_str = '                    /d            \\\nDifferentialOperator|--(f(x)),f(x)|\n                    \\dx           /'
    ucode_str = '                    ⎛d            ⎞\nDifferentialOperator⎜──(f(x)),f(x)⎟\n                    ⎝dx           ⎠'
    assert pretty(d) == ascii_str
    assert upretty(d) == ucode_str
    assert latex(d) == 'DifferentialOperator\\left(\\frac{d}{d x} f{\\left(x \\right)},f{\\left(x \\right)}\\right)'
    sT(d, "DifferentialOperator(Derivative(Function('f')(Symbol('x')), Tuple(Symbol('x'), Integer(1))),Function('f')(Symbol('x')))")
    assert str(b) == 'Operator(B,t,1/2)'
    assert pretty(b) == 'Operator(B,t,1/2)'
    assert upretty(b) == 'Operator(B,t,1/2)'
    assert latex(b) == 'Operator\\left(B,t,\\frac{1}{2}\\right)'
    sT(b, "Operator(Symbol('B'),Symbol('t'),Rational(1, 2))")
    assert str(op) == '|psi><psi|'
    assert pretty(op) == '|psi><psi|'
    assert upretty(op) == '❘ψ⟩⟨ψ❘'
    assert latex(op) == '{\\left|\\psi\\right\\rangle }{\\left\\langle \\psi\\right|}'
    sT(op, "OuterProduct(Ket(Symbol('psi')),Bra(Symbol('psi')))")