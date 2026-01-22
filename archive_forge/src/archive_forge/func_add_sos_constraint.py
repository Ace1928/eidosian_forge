from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
from pyomo.core.base.block import _BlockData
from pyomo.core.kernel.block import IBlock
from pyomo.core.base.suffix import active_import_suffix_generator
from pyomo.core.kernel.suffix import import_suffix_generator
from pyomo.core.expr.numvalue import native_numeric_types, value
from pyomo.core.expr.visitor import evaluate_expression
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.core.base.sos import SOSConstraint
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
import time
import logging
def add_sos_constraint(self, con):
    """Add a single SOS constraint to the solver's model (if supported).

        This will keep any existing model components intact.

        Parameters
        ----------
        con: SOSConstraint

        """
    if self._pyomo_model is None:
        raise RuntimeError('You must call set_instance before calling add_sos_constraint.')
    self._add_sos_constraint(con)