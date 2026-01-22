import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import (ValueWarning,
from statsmodels.tools.validation import (string_like,
def _compute_using_eig(self):
    """
        Eigenvalue decomposition method to compute eigenvalues and eigenvectors
        """
    x = self.transformed_data
    self.eigenvals, self.eigenvecs = np.linalg.eigh(x.T.dot(x))