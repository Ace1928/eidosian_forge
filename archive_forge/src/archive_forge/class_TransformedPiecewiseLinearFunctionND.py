from collections.abc import Sized
import logging
from pyomo.core.kernel.block import block
from pyomo.core.kernel.set_types import IntegerSet
from pyomo.core.kernel.variable import variable, variable_dict, variable_tuple
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.expression import expression, expression_tuple
import pyomo.core.kernel.piecewise_library.util
class TransformedPiecewiseLinearFunctionND(block):
    """Base class for transformed multi-variate piecewise
    linear functions

    A transformed multi-variate piecewise linear functions
    is a block of variables and constraints that enforce a
    piecewise linear relationship between an vector input
    variables and a single output variable.

    Args:
        f (:class:`PiecewiseLinearFunctionND`): The
            multi-variate piecewise linear function to
            transform.
        input: The variable constrained to be the input of
            the piecewise linear function.
        output: The variable constrained to be the output of
            the piecewise linear function.
        bound (str): The type of bound to impose on the
            output expression. Can be one of:

              - 'lb': y <= f(x)
              - 'eq': y  = f(x)
              - 'ub': y >= f(x)
    """

    def __init__(self, f, input=None, output=None, bound='eq'):
        super(TransformedPiecewiseLinearFunctionND, self).__init__()
        assert isinstance(f, PiecewiseLinearFunctionND)
        if bound not in ('lb', 'ub', 'eq'):
            raise ValueError("Invalid bound type %r. Must be one of: ['lb','ub','eq']" % bound)
        self._bound = bound
        self._f = f
        _, ndim = f._tri.points.shape
        if input is None:
            input = [None] * ndim
        self._input = expression_tuple((expression(input[i]) for i in range(ndim)))
        self._output = expression(output)

    @property
    def input(self):
        """The tuple of expressions that store the
        inputs to the piecewise function. The returned
        objects can be updated by assigning to their
        :attr:`expr` attribute."""
        return self._input

    @property
    def output(self):
        """The expression that stores the output of the
        piecewise function. The returned object can be
        updated by assigning to its :attr:`expr`
        attribute."""
        return self._output

    @property
    def bound(self):
        """The bound type assigned to the piecewise
        relationship ('lb','ub','eq')."""
        return self._bound

    @property
    def triangulation(self):
        """The triangulation over the domain of this function"""
        return self._f.triangulation

    @property
    def values(self):
        """The set of values used to defined this function"""
        return self._f.values

    def __call__(self, x):
        """
        Evaluates the piecewise linear function using
        interpolation. This method supports vectorized
        function calls as the interpolation process can be
        expensive for high dimensional data.

        For the case when a single point is provided, the
        argument x should be a (D,) shaped numpy array or
        list, where D is the dimension of points in the
        triangulation.

        For the vectorized case, the argument x should be
        a (n,D)-shaped numpy array.
        """
        return self._f(x)