import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import (ValueWarning,
from statsmodels.tools.validation import (string_like,
def _compute_pca_from_eig(self):
    """
        Compute relevant statistics after eigenvalues have been computed
        """
    vals, vecs = (self.eigenvals, self.eigenvecs)
    indices = np.argsort(vals)
    indices = indices[::-1]
    vals = vals[indices]
    vecs = vecs[:, indices]
    if (vals <= 0).any():
        num_good = vals.shape[0] - (vals <= 0).sum()
        if num_good < self._ncomp:
            import warnings
            warnings.warn('Only {num:d} eigenvalues are positive.  This is the maximum number of components that can be extracted.'.format(num=num_good), EstimationWarning)
            self._ncomp = num_good
            vals[num_good:] = np.finfo(np.float64).tiny
    vals = vals[:self._ncomp]
    vecs = vecs[:, :self._ncomp]
    self.eigenvals, self.eigenvecs = (vals, vecs)
    self.scores = self.factors = self.transformed_data.dot(vecs)
    self.loadings = vecs
    self.coeff = vecs.T
    if self._normalize:
        self.coeff = (self.coeff.T * np.sqrt(vals)).T
        self.factors /= np.sqrt(vals)
        self.scores = self.factors