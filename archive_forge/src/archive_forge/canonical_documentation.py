import abc
import copy
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.utilities import performance_utils as pu
from cvxpy.utilities.deterministic import unique_list
Returns all the atoms present in the args.

        Returns
        -------
        list
        