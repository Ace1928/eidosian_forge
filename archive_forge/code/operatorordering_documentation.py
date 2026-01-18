import warnings
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.physics.quantum import Operator, Commutator, AntiCommutator
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
Normal order an expression with bosonic or fermionic operators. Note
    that this normal order is not equivalent to the original expression, but
    the creation and annihilation operators in each term in expr is reordered
    so that the expression becomes normal ordered.

    Parameters
    ==========

    expr : expression
        The expression to normal order.

    recursive_limit : int (default 10)
        The number of allowed recursive applications of the function.

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger
    >>> from sympy.physics.quantum.boson import BosonOp
    >>> from sympy.physics.quantum.operatorordering import normal_order
    >>> a = BosonOp("a")
    >>> normal_order(a * Dagger(a))
    Dagger(a)*a
    