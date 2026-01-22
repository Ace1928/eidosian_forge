import numpy as np
from scipy import linalg
from scipy.special import xlogy
from scipy.spatial.distance import cdist, pdist, squareform
def _h_gaussian(self, r):
    return np.exp(-(1.0 / self.epsilon * r) ** 2)