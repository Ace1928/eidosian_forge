import abc
from typing import List, Tuple
import numpy as np
from cvxpy.reductions.reduction import Reduction
Returns a stuffed problem.

        The returned problem is a minimization problem in which every
        constraint in the problem has affine arguments that are expressed in
        the form A @ x + b.


        Parameters
        ----------
        problem: The problem to stuff; the arguments of every constraint
            must be affine

        Returns
        -------
        Problem
            The stuffed problem
        InverseData
            Data for solution retrieval
        