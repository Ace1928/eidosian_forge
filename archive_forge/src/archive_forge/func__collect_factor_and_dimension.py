from typing import Dict as tDict, Set as tSet
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.physics.units.dimensions import _QuantityMapper
from sympy.physics.units.quantities import Quantity
from .dimensions import Dimension
def _collect_factor_and_dimension(self, expr):
    """
        Return tuple with scale factor expression and dimension expression.
        """
    from sympy.physics.units import Quantity
    if isinstance(expr, Quantity):
        return (expr.scale_factor, expr.dimension)
    elif isinstance(expr, Mul):
        factor = 1
        dimension = Dimension(1)
        for arg in expr.args:
            arg_factor, arg_dim = self._collect_factor_and_dimension(arg)
            factor *= arg_factor
            dimension *= arg_dim
        return (factor, dimension)
    elif isinstance(expr, Pow):
        factor, dim = self._collect_factor_and_dimension(expr.base)
        exp_factor, exp_dim = self._collect_factor_and_dimension(expr.exp)
        if self.get_dimension_system().is_dimensionless(exp_dim):
            exp_dim = 1
        return (factor ** exp_factor, dim ** (exp_factor * exp_dim))
    elif isinstance(expr, Add):
        factor, dim = self._collect_factor_and_dimension(expr.args[0])
        for addend in expr.args[1:]:
            addend_factor, addend_dim = self._collect_factor_and_dimension(addend)
            if not self.get_dimension_system().equivalent_dims(dim, addend_dim):
                raise ValueError('Dimension of "{}" is {}, but it should be {}'.format(addend, addend_dim, dim))
            factor += addend_factor
        return (factor, dim)
    elif isinstance(expr, Derivative):
        factor, dim = self._collect_factor_and_dimension(expr.args[0])
        for independent, count in expr.variable_count:
            ifactor, idim = self._collect_factor_and_dimension(independent)
            factor /= ifactor ** count
            dim /= idim ** count
        return (factor, dim)
    elif isinstance(expr, Function):
        fds = [self._collect_factor_and_dimension(arg) for arg in expr.args]
        dims = [Dimension(1) if self.get_dimension_system().is_dimensionless(d[1]) else d[1] for d in fds]
        return (expr.func(*(f[0] for f in fds)), *dims)
    elif isinstance(expr, Dimension):
        return (S.One, expr)
    else:
        return (expr, Dimension(1))