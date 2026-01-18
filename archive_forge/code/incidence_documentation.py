from contextlib import nullcontext
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.numvalue import value as pyo_value
from pyomo.repn import generate_standard_repn
from pyomo.util.subsystems import TemporarySubsystemManager
from pyomo.repn.plugins.nl_writer import AMPLRepn
from pyomo.contrib.incidence_analysis.config import (
Get variables that participate in an expression

    The exact variables returned depends on the method used to determine incidence.
    For example, ``method=IncidenceMethod.identify_variables`` will return all
    variables participating in the expression, while
    ``method=IncidenceMethod.standard_repn`` will return only the variables
    identified by ``generate_standard_repn`` which ignores variables that only
    appear multiplied by a constant factor of zero.

    Keyword arguments must be valid options for ``IncidenceConfig``.

    Parameters
    ----------
    expr: ``NumericExpression``
        Expression to search for variables

    Returns
    -------
    list of VarData
        List containing the variables that participate in the expression

    Example
    -------

    .. doctest::

       >>> import pyomo.environ as pyo
       >>> from pyomo.contrib.incidence_analysis import get_incident_variables
       >>> m = pyo.ConcreteModel()
       >>> m.x = pyo.Var([1, 2, 3])
       >>> expr = m.x[1] + 2*m.x[2] + 3*m.x[3]**2
       >>> print([v.name for v in get_incident_variables(expr)])
       ['x[1]', 'x[2]', 'x[3]']
       >>> print([v.name for v in get_incident_variables(expr, linear_only=True)])
       ['x[1]', 'x[2]']

    