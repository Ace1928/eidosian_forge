import numpy as np
from scipy import linalg
from scipy.special import xlogy
from scipy.spatial.distance import cdist, pdist, squareform
def _call_norm(self, x1, x2):
    return cdist(x1.T, x2.T, self.norm)