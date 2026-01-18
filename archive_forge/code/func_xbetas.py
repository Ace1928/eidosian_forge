import numpy as np
import numpy.lib.recfunctions as recf
from scipy import optimize
def xbetas(self, params):
    """these are the V_i
        """
    res = np.empty((self.nobs, self.nchoices))
    for choiceind in range(self.nchoices):
        res[:, choiceind] = np.dot(self.exog_bychoices[choiceind], params[self.beta_indices[choiceind]])
    return res