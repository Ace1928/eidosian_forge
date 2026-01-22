import numpy as np
from scipy import optimize
from statsmodels.regression.linear_model import OLS
def ar1filter(self, xy, alpha):
    return (xy[1:] - alpha * xy[:-1])[self.groups_valid]