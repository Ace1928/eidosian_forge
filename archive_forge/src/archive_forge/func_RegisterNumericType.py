import logging
import sys
from pyomo.common.dependencies import numpy_available
from pyomo.common.deprecation import deprecated, relocated_module_attribute
from pyomo.common.errors import TemplateExpressionError
def RegisterNumericType(new_type: type):
    """Register the specified type as a "numeric type".

    A utility function for registering new types as "native numeric
    types" that can be leaf nodes in Pyomo numeric expressions.  The
    type should be compatible with :py:class:`float` (that is, store a
    scalar and be castable to a Python float).

    Parameters
    ----------
    new_type: type
        The new numeric type (e.g, numpy.float64)

    """
    native_numeric_types.add(new_type)
    native_types.add(new_type)
    nonpyomo_leaf_types.add(new_type)