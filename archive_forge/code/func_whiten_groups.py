import numpy as np
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.tools.grouputils import GroupSorted
def whiten_groups(self, x, cholsigmainv_i):
    wx = whiten_individuals_loop(x, cholsigmainv_i, self.group.group_iter)
    return wx