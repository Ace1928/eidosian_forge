import warnings
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.physics.quantum import Operator, Commutator, AntiCommutator
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
def _normal_order_factor(product, recursive_limit=10, _recursive_depth=0):
    """
    Helper function for normal_order: Normal order a multiplication expression
    with bosonic or fermionic operators. In general the resulting operator
    expression will not be equivalent to original product.
    """
    factors = _expand_powers(product)
    n = 0
    new_factors = []
    while n < len(factors) - 1:
        if isinstance(factors[n], BosonOp) and factors[n].is_annihilation:
            if not isinstance(factors[n + 1], BosonOp):
                new_factors.append(factors[n])
            elif factors[n + 1].is_annihilation:
                new_factors.append(factors[n])
            else:
                if factors[n].args[0] != factors[n + 1].args[0]:
                    new_factors.append(factors[n + 1] * factors[n])
                else:
                    new_factors.append(factors[n + 1] * factors[n])
                n += 1
        elif isinstance(factors[n], FermionOp) and factors[n].is_annihilation:
            if not isinstance(factors[n + 1], FermionOp):
                new_factors.append(factors[n])
            elif factors[n + 1].is_annihilation:
                new_factors.append(factors[n])
            else:
                if factors[n].args[0] != factors[n + 1].args[0]:
                    new_factors.append(-factors[n + 1] * factors[n])
                else:
                    new_factors.append(-factors[n + 1] * factors[n])
                n += 1
        else:
            new_factors.append(factors[n])
        n += 1
    if n == len(factors) - 1:
        new_factors.append(factors[-1])
    if new_factors == factors:
        return product
    else:
        expr = Mul(*new_factors).expand()
        return normal_order(expr, recursive_limit=recursive_limit, _recursive_depth=_recursive_depth + 1)