import numpy as np
from scipy import linalg
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def xpinv(self):
    return linalg.pinv(self.x)