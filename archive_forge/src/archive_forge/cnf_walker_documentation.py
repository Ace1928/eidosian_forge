from pyomo.common import DeveloperError
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import attempt_import
from pyomo.core.expr.logical_expr import special_boolean_atom_types
from pyomo.core.expr.numvalue import native_types, value
from pyomo.core.expr.sympy_tools import (
If the child logical constraints are not literals, substitute
    augmented boolean variables.

    Same return types as to_cnf() function.

    