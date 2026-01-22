import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import (ValueWarning,
from statsmodels.tools.validation import (string_like,
def _compute_eig(self):
    """
        Wrapper for actual eigenvalue method

        This is a workaround to avoid instance methods in __dict__
        """
    if self._method == 'eig':
        return self._compute_using_eig()
    elif self._method == 'svd':
        return self._compute_using_svd()
    else:
        return self._compute_using_nipals()