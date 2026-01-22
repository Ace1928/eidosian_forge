import warnings
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.physics.quantum import Operator, Commutator, AntiCommutator
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
def _normal_ordered_form_factor(product, independent=False, recursive_limit=10, _recursive_depth=0):
    """
    Helper function for normal_ordered_form_factor: Write multiplication
    expression with bosonic or fermionic operators on normally ordered form,
    using the bosonic and fermionic commutation relations. The resulting
    operator expression is equivalent to the argument, but will in general be
    a sum of operator products instead of a simple product.
    """
    factors = _expand_powers(product)
    new_factors = []
    n = 0
    while n < len(factors) - 1:
        if isinstance(factors[n], BosonOp):
            if not isinstance(factors[n + 1], BosonOp):
                new_factors.append(factors[n])
            elif factors[n].is_annihilation == factors[n + 1].is_annihilation:
                if independent and str(factors[n].name) > str(factors[n + 1].name):
                    new_factors.append(factors[n + 1])
                    new_factors.append(factors[n])
                    n += 1
                else:
                    new_factors.append(factors[n])
            elif not factors[n].is_annihilation:
                new_factors.append(factors[n])
            elif factors[n + 1].is_annihilation:
                new_factors.append(factors[n])
            else:
                if factors[n].args[0] != factors[n + 1].args[0]:
                    if independent:
                        c = 0
                    else:
                        c = Commutator(factors[n], factors[n + 1])
                    new_factors.append(factors[n + 1] * factors[n] + c)
                else:
                    c = Commutator(factors[n], factors[n + 1])
                    new_factors.append(factors[n + 1] * factors[n] + c.doit())
                n += 1
        elif isinstance(factors[n], FermionOp):
            if not isinstance(factors[n + 1], FermionOp):
                new_factors.append(factors[n])
            elif factors[n].is_annihilation == factors[n + 1].is_annihilation:
                if independent and str(factors[n].name) > str(factors[n + 1].name):
                    new_factors.append(factors[n + 1])
                    new_factors.append(factors[n])
                    n += 1
                else:
                    new_factors.append(factors[n])
            elif not factors[n].is_annihilation:
                new_factors.append(factors[n])
            elif factors[n + 1].is_annihilation:
                new_factors.append(factors[n])
            else:
                if factors[n].args[0] != factors[n + 1].args[0]:
                    if independent:
                        c = 0
                    else:
                        c = AntiCommutator(factors[n], factors[n + 1])
                    new_factors.append(-factors[n + 1] * factors[n] + c)
                else:
                    c = AntiCommutator(factors[n], factors[n + 1])
                    new_factors.append(-factors[n + 1] * factors[n] + c.doit())
                n += 1
        elif isinstance(factors[n], Operator):
            if isinstance(factors[n + 1], (BosonOp, FermionOp)):
                new_factors.append(factors[n + 1])
                new_factors.append(factors[n])
                n += 1
            else:
                new_factors.append(factors[n])
        else:
            new_factors.append(factors[n])
        n += 1
    if n == len(factors) - 1:
        new_factors.append(factors[-1])
    if new_factors == factors:
        return product
    else:
        expr = Mul(*new_factors).expand()
        return normal_ordered_form(expr, recursive_limit=recursive_limit, _recursive_depth=_recursive_depth + 1, independent=independent)