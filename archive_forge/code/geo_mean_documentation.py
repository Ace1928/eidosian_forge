from typing import List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.errormsg import SECOND_ARG_SHOULD_NOT_BE_EXPRESSION_ERROR_MESSAGE
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import cvxtypes
from cvxpy.utilities.power_tools import (
Returns a shallow copy of the geo_mean atom.

        Parameters
        ----------
        args : list, optional
            The arguments to reconstruct the atom. If args=None, use the
            current args of the atom.

        Returns
        -------
        geo_mean atom
        