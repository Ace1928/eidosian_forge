import numpy as np
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.tools.grouputils import GroupSorted
def whiten_individuals_loop(x, transform, group_iter):
    """apply linear transform for each individual

    loop version
    """
    x_new = []
    for g in group_iter():
        x_g = x[g]
        x_new.append(np.dot(transform, x_g))
    return np.concatenate(x_new)