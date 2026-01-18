from collections.abc import Sized
import logging
from pyomo.core.kernel.block import block
from pyomo.core.kernel.set_types import IntegerSet
from pyomo.core.kernel.variable import variable, variable_dict, variable_tuple
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.expression import expression, expression_tuple
import pyomo.core.kernel.piecewise_library.util

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
        