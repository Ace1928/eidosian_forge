import logging
import math
import itertools
import operator
import types
import enum
from pyomo.common.log import is_debug_set
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.numeric_types import value
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.block import Block, _BlockData
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.constraint import Constraint, ConstraintList
from pyomo.core.base.sos import SOSConstraint
from pyomo.core.base.var import Var, _VarData, IndexedVar
from pyomo.core.base.set_types import PositiveReals, NonNegativeReals, Binary
from pyomo.core.base.util import flatten_tuple
def _characterize_function(name, tol, f_rule, model, points, *index):
    """
    Generates a list of range values and checks
    for convexity/concavity. Assumes domain points
    are sorted in increasing order.
    """
    points = [value(_p) for _p in points]
    if isinstance(f_rule, types.FunctionType):
        values = [f_rule(model, *flatten_tuple((index, x))) for x in points]
    elif f_rule.__class__ is dict:
        if len(index) == 1:
            values = f_rule[index[0]]
        else:
            values = f_rule[index]
    else:
        values = f_rule
    values = [value(_p) for _p in values]
    step = False
    try:
        slopes = [(values[i] - values[i - 1]) / (points[i] - points[i - 1]) for i in range(1, len(points))]
    except ZeroDivisionError:
        step = True
        slopes = [None if points[i] == points[i - 1] else (values[i] - values[i - 1]) / (points[i] - points[i - 1]) for i in range(1, len(points))]
    if not all(itertools.starmap(lambda x1, x2: True if x1 is None or x2 is None else abs(x1 - x2) > tol, zip(slopes, itertools.islice(slopes, 1, None)))):
        msg = "**WARNING: Piecewise component '%s[%s]' has detected slopes of consecutive piecewise segments to be within " + str(tol) + ' of one another. Refer to the Piecewise help documentation for information on how to disable this warning.'
        if index == ():
            index = None
        print(msg % (name, flatten_tuple(index)))
    if step is True:
        return (0, values, True)
    if _isNonDecreasing(slopes):
        return (1, values, False)
    if _isNonIncreasing(slopes):
        return (-1, values, False)
    return (0, values, False)