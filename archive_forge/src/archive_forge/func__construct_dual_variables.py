import abc
import numpy as np
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.expressions import cvxtypes
def _construct_dual_variables(self, args) -> None:
    self.dual_variables = [cvxtypes.variable()(arg.shape) for arg in args]