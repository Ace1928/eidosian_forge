from typing import List, Union
import cvxpy.atoms as atoms
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.transforms import indicator
Combines objectives as log_sum_exp of weighted terms.


    The objective takes the form
        log(sum_{i=1}^n exp(gamma*weights[i]*objectives[i]))/gamma
    As gamma goes to 0, log_sum_exp approaches weighted_sum. As gamma goes to infinity,
    log_sum_exp approaches max.

    Args:
      objectives: A list of Minimize/Maximize objectives.
      weights: A vector of weights.
      gamma: Parameter interpolating between weighted_sum and max.

    Returns:
      A Minimize objective.
    