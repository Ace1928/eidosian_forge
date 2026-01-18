import numpy as np
from scipy import linalg
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def xxthalf(self):
    return np.dot(self.u, self.sdiag)